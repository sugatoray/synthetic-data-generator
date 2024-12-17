# pip install synthetic-dataset-generator
import os

from synthetic_dataset_generator.app import demo

assert os.getenv("HF_TOKEN")
os.environ["BASE_URL"] = "http://127.0.0.1:11434/v1/"
os.environ["MODEL"] = "llama3.1"

demo.launch()