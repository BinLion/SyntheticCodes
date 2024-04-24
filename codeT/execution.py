# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ctypes

from codeT.postprocess import PostProcessor, get_function_name
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
import logging
from codeT._execution import check_correctness_with_test_cases, check_correctness

logging.basicConfig(
    format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

def evaluate_with_test_code(
    samples,
    timeout
):
    logger.info(f'Start evaluation with test code, timeout={timeout}')
    # Check the generated samples against test suites.
    with ProcessPoolExecutor() as executor:

        futures = []
        existed_completion = defaultdict(set)
        results = defaultdict(defaultdict)

        for sample in samples:
            task_id = sample["task_id"]
            prompt = sample['prompt']
            test = sample['test']
            entry_point = sample['entry_point']
            completion = sample["completion"]
            if completion in existed_completion[task_id]:
                continue
            existed_completion[task_id].add(completion)
            args = (task_id, prompt, completion, test, entry_point, timeout)
            future = executor.submit(check_correctness, *args)
            futures.append(future)
        logger.info(f'{len(futures)} execution requests are submitted')
        
        for idx, future in enumerate(as_completed(futures)):
            logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))
            result = future.result()
            results[result["task_id"]][result["completion"]] = result

    logger.info('execution finished! start parsing results')
    samples_with_result = []
    for sample in samples:
        task_id = sample["task_id"]
        completion = sample["completion"]
        result = results[task_id][completion]
        sample["result"] = result["result"]
        sample["passed"] = result["passed"]
        samples_with_result.append(sample)

    assert len(samples_with_result) == len(samples), "Some problems are not attempted."

    return samples_with_result

def evaluate_with_test_cases(
    solutions,
    test_cases_dict,
    timeout,
    limit
):
    logger.info(f'Start evaluation with test cases, timeout={timeout}, limit={limit}')
    # Check the generated solutions against test suites.
    with ProcessPoolExecutor() as executor:
        futures = []
        results_list = []
        existed_completion = defaultdict(set)

        for solution in solutions:
            task_id = solution['task_id']
            prompt = solution['prompt']
            completion = solution['completion']
            if completion in existed_completion[task_id]:
                continue
            existed_completion[task_id].add(completion)
            task_test_cases = test_cases_dict[task_id]
            if not task_test_cases:
                continue
            # get limited test cases
            limited_task_test_cases = [cases_per_sample[:limit] for cases_per_sample in task_test_cases]
            limited_task_test_cases = sum(limited_task_test_cases, [])
            
            args = (task_id, prompt, completion, list(set(limited_task_test_cases)), timeout)
            future = executor.submit(check_correctness_with_test_cases, *args)
            futures.append(future)

        logger.info(f'{len(futures)} execution requests are submitted')
        for idx, future in enumerate(as_completed(futures)):
            logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))
            result = future.result()
            results_list.append(result)

    logger.info('execution finished!')
    return results_list

def best_solution(
    exercise,
    timeout,
    limit,
    debug = False
):
    test_cases_by_task = []
    entry_point = get_function_name(exercise['problem'])
    for sample in exercise['tests']:
        test_cases = PostProcessor.test_case_extract(sample, entry_point)
        test_cases_by_task.append(test_cases)
    
    # print(test_cases_by_task)
    # get limited test cases
    limited_task_test_cases = [cases_per_sample[:limit] for cases_per_sample in test_cases_by_task]
    limited_task_test_cases = sum(limited_task_test_cases, [])
    # print(limited_task_test_cases)
    logger.info(f'{len(exercise["solutions"])} solutions, {len(limited_task_test_cases)} test cases')
    with ProcessPoolExecutor() as executor:
        futures = []
        results_list = []
        for sample in exercise['solutions']:
            completion = PostProcessor.solution_extract(sample)
            args = (exercise['exercise_id'], exercise['problem'], completion, list(set(limited_task_test_cases)), timeout, debug)
            future = executor.submit(check_correctness_with_test_cases, *args)
            futures.append(future)
            
        logger.info(f'{len(futures)} execution requests are submitted')
        for idx, future in enumerate(as_completed(futures)):
            logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))
            result = future.result()
            results_list.append(result)
            
    logger.info('execution finished!')
    # print(results_list)
    return results_list

def pass_most_solution(execution_result):
    max_true_result = None
    max_true_count = 0
    
    for item in execution_result:
        if item['passed'] == False:
            continue
        current_true_count = item['result'].count(True)
        
        if current_true_count > max_true_count:
            max_true_result = item
            max_true_count = current_true_count

    return max_true_result
