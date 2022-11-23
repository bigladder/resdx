import os
from doit.tools import create_folder

OUTPUT_PATH = "output"

def task_test():
  '''Performs tests'''
  return {
    'targets': ['output/heat-pump.png'],
    'actions': [
      (create_folder, [OUTPUT_PATH]),
      'pytest -v test'],
    'clean': True
  }

def task_examples():
  '''Run examples'''
  return {
    'targets': ['output/title24-heat-pump.py',
                'output/cooling-cop-v-seer.png',
                'output/heating-cop-v-hspf.png'],
    'actions': [
      (create_folder, [OUTPUT_PATH]),
      'python examples/model-comparison.py',
      'python examples/model-verification.py',
      'python examples/inverse-calculations.py',
      'python examples/generate-205.py'],
    'clean': True
  }
