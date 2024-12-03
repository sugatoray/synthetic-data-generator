import os
import warnings

import argilla as rg

# Tasks
TEXTCAT_TASK = "text_classification"
SFT_TASK = "supervised_fine_tuning"

# Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError(
        "HF_TOKEN is not set. Ensure you have set the HF_TOKEN environment variable that has access to the Hugging Face Hub repositories and Inference Endpoints."
    )

# Inference
DEFAULT_BATCH_SIZE = 5
MODEL = os.getenv("MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
API_KEYS = (
    [os.getenv("HF_TOKEN")]
    + [os.getenv(f"HF_TOKEN_{i}") for i in range(1, 10)]
    + [os.getenv("API_KEY")]
)
API_KEYS = [token for token in API_KEYS if token]
BASE_URL = os.getenv("BASE_URL", "https://api-inference.huggingface.co/v1/")

if BASE_URL != "https://api-inference.huggingface.co/v1/" and len(API_KEYS) == 0:
    raise ValueError(
        "API_KEY is not set. Ensure you have set the API_KEY environment variable that has access to the Hugging Face Inference Endpoints."
    )
if "Qwen2" not in MODEL and "Llama-3" not in MODEL:
    SFT_AVAILABLE = False
    warnings.warn(
        "SFT_AVAILABLE is set to False because the model is not a Qwen or Llama model."
    )
    MAGPIE_PRE_QUERY_TEMPLATE = None
else:
    SFT_AVAILABLE = True
    if "Qwen2" in MODEL:
        MAGPIE_PRE_QUERY_TEMPLATE = "qwen2"
    else:
        MAGPIE_PRE_QUERY_TEMPLATE = "llama3"

# Embeddings
STATIC_EMBEDDING_MODEL = "minishlab/potion-base-8M"

# Argilla
ARGILLA_API_URL = os.getenv("ARGILLA_API_URL")
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY")
if ARGILLA_API_URL is None or ARGILLA_API_KEY is None:
    ARGILLA_API_URL = os.getenv("ARGILLA_API_URL_SDG_REVIEWER")
    ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY_SDG_REVIEWER")

if ARGILLA_API_URL is None or ARGILLA_API_KEY is None:
    warnings.warn("ARGILLA_API_URL or ARGILLA_API_KEY is not set")
    argilla_client = None
else:
    argilla_client = rg.Argilla(
        api_url=ARGILLA_API_URL,
        api_key=ARGILLA_API_KEY,
    )
