# pip install synthetic-dataset-generator
import os

from synthetic_dataset_generator.app import demo

assert os.getenv("HF_TOKEN")
os.environ["BASE_URL"] = "https://api.openai.com/v1/"
os.environ["API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["MODEL"] = "gpt-4o"

demo.launch()
