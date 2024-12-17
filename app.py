import os

from synthetic_dataset_generator import launch

os.environ["BASE_URL"] = "http://localhost:11434/v1"
os.environ["MODEL"] = "llama3.1"

launch()
