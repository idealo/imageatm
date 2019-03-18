import json
import yaml
from pathlib import Path
from typing import Union


def load_json(file_path: Path) -> Union[dict, list]:
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Union[dict, list], target_file: str):
    with open(target_file, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_yaml(file_path: str) -> str:
    with open(file_path, 'r') as f:
        return yaml.load(f)
