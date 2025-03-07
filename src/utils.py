"""The collection of useful functions.
"""
import os
from pathlib import Path
import pickle
from tabulate import tabulate

def mkdir(path: Path) -> None:
    """Check if the folder exists and create it
    if it does not exist.
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def get_parent_path(lvl: int=0) -> Path:
    """Get the lvl-th parent path as root path.
    Return current file path when lvl is zero.
    Must be called under the same folder.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    if lvl > 0:
        for _ in range(lvl):
            path = os.path.abspath(os.path.join(path, os.pardir))
    return path

def save_file(data: dict, file_nmae: str) -> None:
    """Save data to file.
    """
    root = get_parent_path(lvl=1)
    path = os.path.join(root, 'data', 'excitation_signals')
    mkdir(path)
    path_file = os.path.join(path, file_nmae)
    with open(path_file, 'wb') as file:
        pickle.dump(data, file)

def load_file(file_nmae: str) -> dict:
    """Load data from file.
    """
    root = get_parent_path(lvl=1)
    path_file = os.path.join(root, 'data', 'excitation_signal', file_nmae)
    with open(path_file, 'rb') as file:
        data = pickle.load(file)
    return data

def print_info(**kwargs):
    """Print information on the screen.
    """
    processed_kwargs, key_map = preprocess_kwargs(**kwargs)
    columns = [key_map[key] for key in processed_kwargs.keys()]
    data = list(zip(*processed_kwargs.values()))
    table = tabulate(data, headers=columns, tablefmt="grid")
    print(table)

def preprocess_kwargs(**kwargs):
    """Project the keys.
    """
    replacement_rules = {
        "__slash__": "/",
        "__percent__": "%"
    }

    processed_kwargs = {}
    key_map = {}
    for key, value in kwargs.items():
        new_key = key
        
        for old, new in replacement_rules.items():
            new_key = new_key.replace(old, new)

        processed_kwargs[key] = value
        key_map[key] = new_key
    
    return processed_kwargs, key_map