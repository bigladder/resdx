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
