import random
from typing import List

from distilabel.llms import InferenceEndpointsLLM
from distilabel.steps.tasks import (
    GenerateTextClassificationData,
    TextClassification,
    TextGeneration,
)

from src.distilabel_dataset_generator.pipelines.base import (
    MODEL,
    _get_next_api_key,
)
from src.distilabel_dataset_generator.utils import get_preprocess_labels

PROMPT_CREATION_PROMPT = """You are an AI assistant specialized in generating very precise text classification tasks for dataset creation.

Your task is to write a prompt following the instruction of the user. Respond with the prompt and nothing else.

The prompt you write should follow the same style and structure as the following example prompts, clearly specifying the possible classification labels.

If a label is composed of multiple words, use a hyphen to separate them. For example, 'smartphone-review', 'customer-service', 'product-quality'.:

Classify the following customer review of a cinema as either 'positive' or 'negative'.

Classify the following news article into one or more of the following categories: 'politics', 'sports', 'technology', 'entertainment', 'health', 'business', 'environment', 'education', 'science', 'international'.

Determine the sentiment of the following social media post: 'ambiguous', 'sarcastic', 'informative', 'emotional'.

Identify the issue category for the following technical support ticket: 'billing', 'technical', 'account', 'shipping', 'returns', 'installation', 'subscription'.

Classify the following movie review into one of the following categories: 'critical', 'praise', 'disappointed', 'enthusiastic'.

Determine the level of customer satisfaction from the following customer service transcript: 'satisfied', 'dissatisfied', 'highly-satisfied', 'somewhat-dissatisfied', 'indifferent'.

Categorize the following product description into one of the following product types: 'smartphone', 'laptop', 'tablet', 'smartwatch', 'e-reader', 'headphones'.

Classify the following tweet as expressing either 'support' or 'opposition' to the political event discussed.

Classify the following restaurant review into one of the following categories: 'food-quality', 'service', 'ambiance', or 'price'.

Classify the following blog post based on its primary fashion trend or style: 'casual', 'formal', 'streetwear', 'vintage' or 'sustainable-fashion'.

User dataset description:
"""

DEFAULT_DATASET_DESCRIPTIONS = [
    "A dataset covering customer reviews for an e-commerce website.",
    "A dataset covering news articles about various topics.",
]


def generate_pipeline_code(
    system_prompt: str,
    difficulty: str = None,
    clarity: str = None,
    labels: List[str] = None,
    num_labels: int = 1,
    num_rows: int = 10,
) -> str:
    labels = get_preprocess_labels(labels)
    base_code = f"""
# Requirements: `pip install distilabel[hf-inference-endpoints]`
import os
import random
from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, KeepColumns
from distilabel.steps.tasks import {"GenerateTextClassificationData" if num_labels == 1 else "GenerateTextClassificationData, TextClassification"}

MODEL = "{MODEL}"
TEXT_CLASSIFICATION_TASK = "{system_prompt}"
os.environ["HF_TOKEN"] = (
    "hf_xxx"  # https://huggingface.co/settings/tokens/new?ownUserPermissions=repo.content.read&ownUserPermissions=repo.write&globalPermissions=inference.serverless.write&canReadGatedRepos=true&tokenType=fineGrained
)

with Pipeline(name="textcat") as pipeline:

    task_generator = LoadDataFromDicts(data=[{{"task": TEXT_CLASSIFICATION_TASK}}])

    textcat_generation = GenerateTextClassificationData(
        llm=InferenceEndpointsLLM(
            model_id=MODEL,
            tokenizer_id=MODEL,
            api_key=os.environ["HF_TOKEN"],
            generation_kwargs={{
                "temperature": 0.8,
                "max_new_tokens": 2048,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95,
            }},
        ),
        seed=random.randint(0, 2**32 - 1),
        difficulty={None if difficulty == "mixed" else repr(difficulty)},
        clarity={None if clarity == "mixed" else repr(clarity)},
        num_generations={num_rows},
        output_mappings={{"input_text": "text"}},
    )
    """

    if num_labels == 1:
        return (
            base_code
            + """
    keep_columns = KeepColumns(
        columns=["text", "label"],
    )

    # Connect steps in the pipeline
    task_generator >> textcat_generation >> keep_columns

    if __name__ == "__main__":
        distiset = pipeline.run()
    """
        )

    return (
        base_code
        + f"""
    keep_columns = KeepColumns(
        columns=["text"],
    )

    textcat_labeller = TextClassification(
        llm=InferenceEndpointsLLM(
            model_id=MODEL,
            tokenizer_id=MODEL,
            api_key=os.environ["HF_TOKEN"],
            generation_kwargs={{
                "temperature": 0.8,
                "max_new_tokens": 2048,
            }},
        ),
        n={num_labels},
        available_labels={labels},
        context=TEXT_CLASSIFICATION_TASK,
        default_label="unknown"
    )

    # Connect steps in the pipeline
    task_generator >> textcat_generation >> keep_columns >> textcat_labeller

    if __name__ == "__main__":
        distiset = pipeline.run()
    """
    )


def get_textcat_generator(difficulty, clarity, is_sample):
    textcat_generator = GenerateTextClassificationData(
        llm=InferenceEndpointsLLM(
            model_id=MODEL,
            tokenizer_id=MODEL,
            api_key=_get_next_api_key(),
            generation_kwargs={
                "temperature": 0.9,
                "max_new_tokens": 256 if is_sample else 2048,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95,
            },
        ),
        difficulty=None if difficulty == "mixed" else difficulty,
        clarity=None if clarity == "mixed" else clarity,
        seed=random.randint(0, 2**32 - 1),
    )
    textcat_generator.load()
    return textcat_generator


def get_labeller_generator(system_prompt, labels, num_labels):
    labeller_generator = TextClassification(
        llm=InferenceEndpointsLLM(
            model_id=MODEL,
            tokenizer_id=MODEL,
            api_key=_get_next_api_key(),
            generation_kwargs={
                "temperature": 0.7,
                "max_new_tokens": 2048,
            },
        ),
        context=system_prompt,
        available_labels=labels,
        n=num_labels,
        default_label="unknown",
    )
    labeller_generator.load()
    return labeller_generator


def get_prompt_generator():
    prompt_generator = TextGeneration(
        llm=InferenceEndpointsLLM(
            api_key=_get_next_api_key(),
            model_id=MODEL,
            tokenizer_id=MODEL,
            generation_kwargs={
                "temperature": 0.8,
                "max_new_tokens": 2048,
                "do_sample": True,
            },
        ),
        use_system_prompt=True,
    )
    prompt_generator.load()
    return prompt_generator
