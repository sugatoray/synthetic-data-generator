from src.distilabel_dataset_generator import HF_TOKENS

DEFAULT_BATCH_SIZE = 5
TOKEN_INDEX = 0
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def _get_next_api_key():
    global TOKEN_INDEX
    api_key = HF_TOKENS[TOKEN_INDEX % len(HF_TOKENS)]
    TOKEN_INDEX += 1
    return api_key
