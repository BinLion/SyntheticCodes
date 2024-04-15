from typing import List, Union
import os
from pathlib import Path

from dataset_gen.dataset_gen import Exercise, ExerciseSolutions, ExerciseTests


def load_one_file(path: Union[Path, str]) -> List[Exercise]:
    with open(path, "r") as f:
        lines = f.readlines()
    return [Exercise.parse_raw(line) for line in lines]


def load_all_exo(path: Union[Path, str]) -> List[Exercise]:
    if isinstance(path, str):
        path = Path(path)
    exos: List[Exercise] = []
    for sub_dir in os.listdir(path):
        for fn in os.listdir(path / sub_dir):
            exos += load_one_file(path / sub_dir / fn)
    return exos

def load_all_solutions(path: Union[Path, str]) -> List[ExerciseSolutions]:
    if isinstance(path, str):
        path = Path(path)
    solutions: List[ExerciseSolutions] = []
    for file in os.listdir(path):
        with open(path / file, "r") as f:
            lines = f.readlines()

        sls = [ExerciseSolutions.parse_raw(line) for line in lines]
        for s in sls:
            good_samples = []
            for sample in s.solutions:
                if filter_syntax_check(s.problem + sample):
                    good_samples.append(sample)
            if len(good_samples) > 0:
                s.solutions = good_samples
                solutions.append(s)
    
    return solutions

def load_all_tests(path: Union[Path, str]) -> List[ExerciseTests]:
    if isinstance(path, str):
        path = Path(path)
    cases: List[ExerciseTests] = []
    for file in os.listdir(path):
        with open(path / file, "r") as f:
            lines = f.readlines()

        sls = [ExerciseTests.parse_raw(line) for line in lines]
        for s in sls:
            good_samples = []
            for sample in s.tests:
                if filter_syntax_check(s.problem + sample):
                    good_samples.append(sample)
            if len(good_samples) > 0:
                s.tests = good_samples
                cases.append(s)
    
    return cases


def filter_bad_exos(
    exos: List[Exercise], carac_to_remove=["??", "___", "Docstring explaining the exercise", "name(args)"]
) -> List[Exercise]:
    clean_exos: List[Exercise] = []
    for exo in exos:
        keep = True
        
        if filter_syntax_check(exo.problem + exo.solution) == False:
            keep = False
            continue
        
        for carac in carac_to_remove:
            if carac in exo.solution or carac in exo.problem:
                keep = False
                break

        if keep:
            clean_exos.append(exo)

    return clean_exos

def filter_syntax_check(code: str) -> bool:
    # return True
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError as e: 
        # print the error message
        print(f"Syntax error in {code}")
        return False


def remove_extra(exos: List[Exercise], carac_to_split=["# Test", "```"]):
    for exo in exos:
        for carac in carac_to_split:
            exo.solution = exo.solution.split(carac)[0]

def deduplicate(exos: List[Exercise]):
    seen = set()
    deduplicated = [item for item in exos if item.exercise_id not in seen and not seen.add(item.exercise_id)]

    return deduplicated


def load_and_filter_exos(path: Union[Path, str]) -> List[Exercise]:
    exos = load_all_exo(path)
    print(f"all: {len(exos)}")

    clean_exos = filter_bad_exos(exos)
    remove_extra(clean_exos)
    print(f"after filter: {len(clean_exos)}")

    clean_exos = deduplicate(clean_exos)
    print(f"after dedup: {len(clean_exos)}")
    
    return clean_exos

    
