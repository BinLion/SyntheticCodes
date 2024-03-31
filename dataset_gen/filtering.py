from typing import List, Union
import os
from pathlib import Path

from dataset_gen.dataset_gen import Exercise


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


def filter_bad_exos(
    exos: List[Exercise], carac_to_remove=["??", "___", "Docstring explaining the exercise", "name(args)"]
) -> List[Exercise]:
    clean_exos: List[Exercise] = []
    for exo in exos:
        keep = True
        
        if filter_syntax_check(exo) == False:
            keep = False
            continue
        
        for carac in carac_to_remove:
            if carac in exo.solution or carac in exo.problem:
                keep = False
                break

        if keep:
            clean_exos.append(exo)

    return clean_exos

def filter_syntax_check(exo: Exercise) -> bool:
    try:
        compile(exo.problem + exo.solution, "<string>", "exec")
        return True
    except SyntaxError as e: 
        # print the error message
        print(f"Syntax error in {exo}. msg: {e.msg}")
        return False


def remove_extra(exos: List[Exercise], carac_to_split=["# Test", "```"]):
    for exo in exos:
        for carac in carac_to_split:
            exo.solution = exo.solution.split(carac)[0]


def load_and_filter_exos(path: Union[Path, str]) -> List[Exercise]:
    exos = load_all_exo(path)
    print(len(exos))
    clean_exos = filter_bad_exos(exos)
    print(len(clean_exos))

    remove_extra(clean_exos)
    return clean_exos
