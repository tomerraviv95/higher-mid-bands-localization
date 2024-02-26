import os

# main folders
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
CODE_DIR = os.path.join(ROOT_DIR, 'python_code')
RESOURCES_DIR = os.path.join(ROOT_DIR, 'resources')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

# subfolders
CONFIG_PATH = os.path.join(CODE_DIR, 'config.yaml')
RAYTRACING_DIR = os.path.join(RESOURCES_DIR, 'raytracing')
NY_DIR = os.path.join(RESULTS_DIR, 'ny')
