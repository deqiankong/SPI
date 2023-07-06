import os
import json
from tqdm import tqdm

def load_jsons(path):
    data = []
    with open(path, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, total=len(lines), desc="Read JSONS data"):
        data.append(json.loads(line))
    return data