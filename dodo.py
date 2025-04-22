from pathlib import Path
from doit.tools import create_folder

OUTPUT_PATH = "output"
create_folder(OUTPUT_PATH)


def task_test():
    """Performs tests"""
    return {
        "actions": ["pytest -v test", 'echo "Tests completed"'],
        "clean": True,
        "verbosity": 2,
    }


def task_examples():
    """Run examples"""
    print("Running examples...")
    for file_path in Path("examples").glob("*.py"):
        yield {
            "name": file_path.name,
            "actions": ['echo "Example starting..."', f"python {file_path}"],
            "clean": True,
            "verbosity": 2,
        }
