import os
from doit.tools import create_folder

OUTPUT_PATH = "output"

def task_test():
  '''Performs tests'''
  return {
    'targets': ['output/heat-pump.py'],
    'actions': [
      (create_folder, [OUTPUT_PATH]),
      'pytest -v test'],
    'clean': True
  }

def task_examples():
  '''Run examples'''
  return {
    'targets': ['output/title24-heat-pump.py'],
    'actions': [
      (create_folder, [OUTPUT_PATH]),
      'python examples/model-comparison.py'],
    'clean': True
  }
