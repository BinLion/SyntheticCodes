import itertools
import json
import os
from pathlib import Path
from typing import List
import pandas as pd
from typer import Typer
from rich.progress import track

from codeT.execution import best_solution, pass_most_solution
from dataset_gen.create_prompts import Query, Topic, create_prompts, create_prompt_query
from dataset_gen.dataset_gen import FalconGenerator, MonkeyGenerator, load_exercises, load_leaves, load_prompts, mass_generation, mass_solutions_generation, mass_tests_generation, write_results_to_jsonl
from dataset_gen.filtering import load_all_solutions, load_all_tests, load_and_filter_exos, load_solutions_with_tests, read_jsonl, merge_dicts, write_jsonl
from falcon.TextGenerationInference import TGI, GenerateParameters, GenerateRequest

app = Typer()

@app.command()
def generate(
    prompt_path: str,
    output_path: str,
    endpoint: str,
    region: str = "us-west-2",
    debug: bool = False,
    debug_speed: int = 2,
    retries: int = 5,
    pool_size: int = 2,
    n_prompts: int = 0,
    
):
    prompts = load_prompts(prompt_path)
    prompts_selection = [i.query for i in prompts]
    print(f"prompts: {len(prompts_selection)}")
    solo_prompts = list(set(prompts_selection))
    print(f"solo prompts: {len(solo_prompts)}")
    if n_prompts > 0: 
        solo_prompts = solo_prompts[:n_prompts]
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not debug:
        def get_generator():
            return FalconGenerator(endpoint, region)
    else:
        def get_generator():
            return MonkeyGenerator(speed=debug_speed)
    
    mass_generation(
        solo_prompts,
        get_generator,
        save_dir=output_path,
        pool_size=pool_size,
        retries=retries,
    )
    
    
@app.command()
def prompts(
    leaves_path: str = "dataset_gen/tree/subsubtopics.json",
    debug: bool = True
):
    if debug:
        n_combinations = 2
    else:
        n_combinations = 200
    
    leaves = leaves = load_leaves(leaves_path)

    with open("dataset_gen/tree/professions.json", "r") as openfile:
        professions = list(json.load(openfile))
        
    prompts: List[List[Query]] = [
        create_prompts(
            i,
            combination_options=leaves,
            professions=professions,
            n=n_combinations,
        )
        for i in leaves
    ]

    prompts_list = list(itertools.chain(*prompts))
    prompts_json = json.dumps([p.dict() for p in prompts_list])
    with open("dataset_gen/tree/prompts.json", "w") as outfile:
        outfile.write(prompts_json)

@app.command()
def filter(exo_path: Path, dataset_file: str):
    print(exo_path)
    exos = load_and_filter_exos(exo_path)
    write_results_to_jsonl(dataset_file, exos)

@app.command()
def filter_solutions(solutions_path: Path, dataset_file: str):
    print(solutions_path)
    items = load_all_solutions(solutions_path)
    write_results_to_jsonl(dataset_file, items)

@app.command()
def filter_tests(tests_path: Path, dataset_file: str):
    print(tests_path)
    items = load_all_tests(tests_path)
    write_results_to_jsonl(dataset_file, items)

@app.command()
def solutions(exercise_path: Path,
              output_path: str,
              n_samples: int,
              endpoint: str,
              region: str = "us-west-2",
              debug: bool = False,
              debug_speed: int = 2,
              pool_size: int = 8,
              retries: int = 5,
):
    exercises = load_exercises(exercise_path)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not debug:
        def get_generator():
            return FalconGenerator(endpoint, region)
    else:
        def get_generator():
            return MonkeyGenerator(speed=debug_speed)
    
    mass_solutions_generation(
        exercises,
        get_generator,
        save_dir=output_path,
        pool_size=pool_size,
        retries=retries,
        n_solutions=n_samples,
    )
    
@app.command()
def tests(exercise_path: Path,
              output_path: str,
              n_samples: int,
              endpoint: str,
              region: str = "us-west-2",
              debug: bool = False,
              debug_speed: int = 2,
              pool_size: int = 8,
              retries: int = 5,
):
    exercises = load_exercises(exercise_path)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not debug:
        def get_generator():
            return FalconGenerator(endpoint, region)
    else:
        def get_generator():
            return MonkeyGenerator(speed=debug_speed)
    
    mass_tests_generation(
        exercises,
        get_generator,
        save_dir=output_path,
        pool_size=pool_size,
        retries=retries,
        n_solutions=n_samples,
    )

@app.command()
def merge(
    solutions_path: Path,
    test_cases_path: Path,
    output_path: str      
):
    solutions = read_jsonl(solutions_path)
    tests = read_jsonl(test_cases_path)
    merged_data = merge_dicts(tests, solutions)
    write_jsonl(merged_data, output_path)
    

@app.command()
def codet(
    data_path: Path,
    output_path: str
):
    data = load_solutions_with_tests(data_path)
    dataset = []
    for item in data:
        result = best_solution(item, 0.5, 5)
        best = pass_most_solution(result)
        if best is None:
            print(f"exercise {item['exercise_id']} solutions failed all test cases")
            continue
        dataset.append({"exercise_id": best['task_id'], "problem": best['prompt'], "solution": best['completion']})

    with open(output_path, 'w') as file:
        for item in dataset:
            file.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    app()