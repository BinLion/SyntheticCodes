import itertools
import json
import os
from pathlib import Path
from typing import List
import pandas as pd
from typer import Typer
from rich.progress import track

from dataset_gen.create_prompts import Query, Topic, create_prompts, create_prompt_query
from dataset_gen.dataset_gen import FalconGenerator, MonkeyGenerator, load_leaves, load_prompts, mass_generation, write_results_to_jsonl
from dataset_gen.filtering import load_and_filter_exos
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
            return FalconGenerator(endpoint)
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
    print(len(exos))
    write_results_to_jsonl(dataset_file, exos)

if __name__ == "__main__":
    app()