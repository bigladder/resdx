from pathlib import Path
from doit.tools import create_folder

OUTPUT_PATH = "output"
create_folder(OUTPUT_PATH)


def task_test():
    """Performs tests"""
    return {
        "actions": ["pytest -v test"],
        "clean": True,
    }


def task_examples():
    """Run examples"""
    for file_path in Path("examples").glob("*.py"):
        yield {
            "name": file_path.name,
            "actions": [f"python {file_path}"],
            "clean": True,
        }
