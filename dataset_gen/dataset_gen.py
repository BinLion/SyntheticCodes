

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import random
import threading
import time
from typing import Callable, List, Protocol
from openai import OpenAIError
from pydantic import BaseModel

from dataset_gen.create_prompts import Topic, Query
from rich.progress import (
    Progress,
    TimeElapsedColumn,
    TextColumn,
)

from falcon.TextGenerationInference import TGI, GenerateParameters, GenerateRequest

THREAD_LOCK = threading.Lock()

class Exercise(BaseModel):
    exercise_id: str
    problem: str
    solution: str


class Result(BaseModel):
    prompt: str
    output: str
    
class Generator(Protocol):
    def generate(self, prompt: str) -> Result:
        ...
        
class GenerationError(OpenAIError):
    ...
    
class MonkeyGenerator:
    """
    A generator with a random response time and a random failure rate
    """

    def __init__(self, speed: int = 2, n_functions: int = 10):
        self.speed = speed
        self.n_functions = n_functions

    def generate(self, prompt: str) -> Result:
        seed = random.randint(0, 100)

        if self.speed > 0:
            time.sleep(seed / 100 * self.speed)
        # if not (seed % 50):
        #     raise GenerationError("Monkey failed")
        return Result(
            prompt=prompt,
            output='def gorilla(): """Empty function for a gorilla""" return 0'
            * self.n_functions,
        )

class FalconGenerator:
    def __init__(
        self,
        endpoint: str = "40B-code-exercises-2024-03-07-07-30-02",
        region: str = "us-west-2"
    ):
        self.endpoint = endpoint
        self.region = region

    def generate(self, prompt: str) -> Result:
        model = TGI(endpoint_name=self.endpoint, region_name=self.region)
        # stop_words = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"]
        stop_words = ["\ndef", "\n#", "\n```"]
        # stop_regex = re.compile("|".join(map(re.escape, stop_words)))

        params = GenerateParameters( max_new_tokens=512, 
                            temperature=1, 
                            stop =stop_words, 
                            top_p = 0.95,
                            #return_log_probs = True,
                            )

        req = GenerateRequest(prompt + "\ndef", params)
        _outputs = model.sm_query(req)
        print(prompt + "\n")
        print(f"completion: \n{_outputs[0]['generated_text']}")

        result = Result(
            prompt=prompt, output= "def" + _outputs[0]["generated_text"]
        )

        return result

def mass_generation(
    prompts: List[str],
    get_generator: Callable[[], Generator],
    save_dir: str,
    pool_size: int = 10,
    retries: int = 10,
):
    """
    Generate from a list of prompts. Use a thread pool to parallelize the generation with catch and retry mechanism
    """
    with Progress(
        *Progress.get_default_columns(),
        "•",
        TimeElapsedColumn(),
    ) as progress:
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            progress_task = progress.add_task(
                "[red]Generating...",
                total=len(prompts),
            )

            def update_progress():
                progress.update(
                    progress_task,
                    advance=1,
                )

            tasks = []

            for prompt in prompts:
                tasks.append(
                    executor.submit(
                        _generation_wrapper,
                        prompt,
                        get_generator,
                        update_progress,
                        save_dir,
                        retries,
                    )
                )

            for task in tasks:
                try:
                    task.result()
                except Exception as e:
                    print(e)

def generation(
    prompt: str,
    generator: Generator,
    update_progress: Callable,
    retries: int,
) -> List[Exercise]:
    success = False
    time.sleep(random.random())
    for i in range(retries):
        try:
            result = generator.generate(prompt)
            success = True
        except GenerationError:
            print(f"Generation failed for prompt {prompt}, retrying {i + 1}/{retries}")
            time.sleep(1)
        else:
            break

    if success:
        exercises = generator_to_exercises(result.output)
        update_progress()
        return exercises

    else:
        print(f"Generation failed for prompt {prompt}, skipping")
        return [Exercise(exercise_id="", problem=prompt, solution="")]


def _generation_wrapper(
    prompt: str,
    get_generator: Callable[[], Generator],
    update_progress: Callable,
    save_dir: str,
    retries: int,
):
    file_path_sum = hashlib.md5(prompt.encode("utf-8")).hexdigest()

    dir_path, file_path = file_path_sum[:4], file_path_sum[4:]
    dir_path = os.path.join(save_dir, dir_path)
    file_path = os.path.join(dir_path, file_path + ".jsonl")

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if os.path.exists(file_path):  # we don't regenerate each query
        print(f"skip {file_path} generation because it already exist ")
        return

    generator = get_generator()

    results = generation(prompt, generator, update_progress, retries)

    write_results_to_jsonl(file_path, results)
                    
def load_leaves(file: str) -> List[Topic]:
    with open(file, "r") as f:
        lines = json.load(f)
    topics = [Topic.parse_obj(line) for line in lines]
    return topics

def load_prompts(file: str) -> List[Query]:
    with open(file, "r") as f:
        lines = json.load(f)
    prompts = [Query.parse_obj(line) for line in lines]
    return prompts

def generator_to_exercises(output: str) -> List[Exercise]:
    exercises = split_exercises(output)
    exercises = [i for i in exercises if check_exercise(i)]
    results = []
    for j in exercises:
        try:
            splitted_exercise = j.split('"""')
            question = '"""'.join(splitted_exercise[:2]) + '"""'
            answer = splitted_exercise[2].strip("```")
            exercise_id = hashlib.md5(question.encode("utf-8")).hexdigest()
            results.append(Exercise(exercise_id=exercise_id, problem=question, solution=answer))
        except IndexError:
            splitted_exercise = j.split("'''")
            question = "'''".join(splitted_exercise[:2]) + "'''"
            answer = splitted_exercise[2].strip("```")
            exercise_id = hashlib.md5(question.encode("utf-8")).hexdigest()
            results.append(Exercise(exercise_id=exercise_id, problem=question, solution=answer))

    return results

def split_exercises(output: str) -> List[str]:
    """Split the result of the generation into separate functions"""
    return ["def" + i for i in output.split("def")[1:]]


def check_exercise(exercise: str) -> bool:
    try:
        if (
            "return" not in exercise.split('"""')[2]
            and "print" not in exercise.split('"""')[2]
        ):
            return False
        else:
            return True
    except IndexError:
        return False

def write_results_to_jsonl(file_path: str, results: List[Exercise]):
    with open(file_path, "w") as file:
        for item in results:
            json.dump(item.dict(), file)
            file.write("\n")