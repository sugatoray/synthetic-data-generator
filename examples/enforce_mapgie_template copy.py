# pip install synthetic-dataset-generator
import os

from synthetic_dataset_generator import launch

os.environ["MAGPIE_PRE_QUERY_TEMPLATE"] = "llama3"
os.environ["MODEL"] = "my_custom_model_trained_on_llama3"

launch()
