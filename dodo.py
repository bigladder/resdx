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
    'targets': ['output/Gross(fannotincluded)COP(atAconditions)_vs_SEER.png','output/Gross(fannotincluded)COP(atH1conditions)_vs_HSPF'],
    'actions': [
      (create_folder, [OUTPUT_PATH]),
      'python examples/inverse-calculations.py'],
    'clean': True
  }
