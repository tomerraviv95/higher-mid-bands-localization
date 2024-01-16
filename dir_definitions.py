import os

# main folders
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
CODE_DIR = os.path.join(ROOT_DIR, 'python_code')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
# subfolders
CONFIG_PATH = os.path.join(CODE_DIR, 'config.yaml')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')