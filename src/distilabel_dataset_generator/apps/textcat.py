import re
import uuid
from typing import List, Union

import argilla as rg
import gradio as gr
import pandas as pd
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from distilabel.distiset import Distiset
from huggingface_hub import HfApi

from src.distilabel_dataset_generator.apps.base import (
    get_argilla_client,
    get_pipeline_code_ui,
    hide_success_message,
    show_success_message_hub,
    validate_argilla_user_workspace_dataset,
    validate_push_to_hub,
)
from src.distilabel_dataset_generator.pipelines.base import (
    DEFAULT_BATCH_SIZE,
)
from src.distilabel_dataset_generator.pipelines.embeddings import (
    get_embeddings,
    get_sentence_embedding_dimensions,
)
from src.distilabel_dataset_generator.pipelines.textcat import (
    DEFAULT_DATASET_DESCRIPTIONS,
    PROMPT_CREATION_PROMPT,
    generate_pipeline_code,
    get_labeller_generator,
    get_prompt_generator,
    get_textcat_generator,
)
from src.distilabel_dataset_generator.utils import (
    get_org_dropdown,
    get_preprocess_labels,
)


def generate_system_prompt(dataset_description, progress=gr.Progress()):
    progress(0.0, desc="Generating text classification task")
    progress(0.3, desc="Initializing text generation")
    generate_description = get_prompt_generator()
    progress(0.7, desc="Generating text classification task")
    system_prompt = next(
        generate_description.process(
            [
                {
                    "system_prompt": PROMPT_CREATION_PROMPT,
                    "instruction": dataset_description,
                }
            ]
        )
    )[0]["generation"]
    progress(1.0, desc="Text classification task generated")
    return system_prompt, pd.DataFrame()


def generate_sample_dataset(system_prompt, progress=gr.Progress()):
    df = generate_dataset(
        system_prompt=system_prompt,
        difficulty="mixed",
        clarity="mixed",
        labels=[],
        num_labels=1,
        num_rows=10,
        progress=progress,
        is_sample=True,
    )
    if "label" in df.columns:
        df = df[["label", "text"]]
    elif "labels" in df.columns:
        df = df[["labels", "text"]]
    return df


def generate_dataset(
    system_prompt: str,
    difficulty: str,
    clarity: str,
    labels: List[str] = None,
    num_labels: int = 1,
    num_rows: int = 10,
    is_sample: bool = False,
    progress=gr.Progress(),
) -> pd.DataFrame:
    if is_sample:
        multiplier = 1
    else:
        multiplier = 2
    progress(0.0, desc="(1/2) Generating text classification data")
    labels = get_preprocess_labels(labels)
    textcat_generator = get_textcat_generator(
        difficulty=difficulty, clarity=clarity, is_sample=is_sample
    )
    labeller_generator = get_labeller_generator(
        system_prompt=system_prompt,
        labels=labels,
        num_labels=num_labels,
    )
    total_steps: int = num_rows * 2
    batch_size = DEFAULT_BATCH_SIZE

    # create text classification data
    n_processed = 0
    textcat_results = []
    while n_processed < num_rows:
        progress(
            multiplier * 0.5 * n_processed / num_rows,
            total=total_steps,
            desc="(1/2) Generating text classification data",
        )
        remaining_rows = num_rows - n_processed
        batch_size = min(batch_size, remaining_rows)
        inputs = [{"task": system_prompt} for _ in range(batch_size)]
        batch = list(textcat_generator.process(inputs=inputs))
        textcat_results.extend(batch[0])
        n_processed += batch_size
    for result in textcat_results:
        result["text"] = result["input_text"]

    # label text classification data
    progress(multiplier * 0.5, desc="(1/2) Generating text classification data")
    if not is_sample:
        n_processed = 0
        labeller_results = []
        while n_processed < num_rows:
            progress(
                0.5 + 0.5 * n_processed / num_rows,
                total=total_steps,
                desc="(1/2) Labeling text classification data",
            )
            batch = textcat_results[n_processed : n_processed + batch_size]
            labels_batch = list(labeller_generator.process(inputs=batch))
            labeller_results.extend(labels_batch[0])
            n_processed += batch_size
        progress(
            1,
            total=total_steps,
            desc="(2/2) Creating dataset",
        )

    # create final dataset
    distiset_results = []
    source_results = textcat_results if is_sample else labeller_results
    for result in source_results:
        record = {
            key: result[key]
            for key in ["text", "label" if is_sample else "labels"]
            if key in result
        }
        distiset_results.append(record)

    dataframe = pd.DataFrame(distiset_results)
    if not is_sample:
        if num_labels == 1:
            dataframe = dataframe.rename(columns={"labels": "label"})
            dataframe["label"] = dataframe["label"].apply(
                lambda x: x.lower().strip() if x.lower().strip() in labels else None
            )
        else:
            dataframe["labels"] = dataframe["labels"].apply(
                lambda x: (
                    list(
                        set(
                            label.lower().strip()
                            for label in x
                            if label.lower().strip() in labels
                        )
                    )
                    if isinstance(x, list)
                    else None
                )
            )
    progress(1.0, desc="Dataset generation completed")
    return dataframe


def push_dataset_to_hub(
    dataframe: pd.DataFrame,
    org_name: str,
    repo_name: str,
    num_labels: int = 1,
    labels: List[str] = None,
    oauth_token: Union[gr.OAuthToken, None] = None,
    private: bool = False,
):
    repo_id = validate_push_to_hub(org_name, repo_name)
    labels = get_preprocess_labels(labels)
    if num_labels == 1:
        dataframe["label"] = dataframe["label"].replace("", None)
        features = Features(
            {"text": Value("string"), "label": ClassLabel(names=labels)}
        )
    else:
        features = Features(
            {
                "text": Value("string"),
                "labels": Sequence(feature=ClassLabel(names=labels)),
            }
        )
    distiset = Distiset({"default": Dataset.from_pandas(dataframe, features=features)})
    distiset.push_to_hub(
        repo_id=repo_id,
        private=private,
        include_script=False,
        token=oauth_token.token,
        create_pr=False,
    )


def push_dataset_to_argilla(
    org_name: str,
    repo_name: str,
    system_prompt: str,
    difficulty: str,
    clarity: str,
    num_labels: int = 1,
    n_rows: int = 10,
    labels: List[str] = None,
    private: bool = False,
    oauth_token: Union[gr.OAuthToken, None] = None,
    progress=gr.Progress(),
) -> pd.DataFrame:
    dataframe = generate_dataset(
        system_prompt=system_prompt,
        difficulty=difficulty,
        clarity=clarity,
        num_labels=num_labels,
        labels=labels,
        num_rows=n_rows,
    )
    push_dataset_to_hub(
        dataframe, org_name, repo_name, num_labels, labels, oauth_token, private
    )
    dataframe = dataframe[
        (dataframe["text"].str.strip() != "") & (dataframe["text"].notna())
    ]
    try:
        progress(0.1, desc="Setting up user and workspace")
        client = get_argilla_client()
        hf_user = HfApi().whoami(token=oauth_token.token)["name"]
        labels = get_preprocess_labels(labels)
        settings = rg.Settings(
            fields=[
                rg.TextField(
                    name="text",
                    description="The text classification data",
                    title="Text",
                ),
            ],
            questions=[
                (
                    rg.LabelQuestion(
                        name="label",
                        title="Label",
                        description="The label of the text",
                        labels=labels,
                    )
                    if num_labels == 1
                    else rg.MultiLabelQuestion(
                        name="labels",
                        title="Labels",
                        description="The labels of the conversation",
                        labels=labels,
                    )
                ),
            ],
            metadata=[
                rg.IntegerMetadataProperty(name="text_length", title="Text Length"),
            ],
            vectors=[
                rg.VectorField(
                    name="text_embeddings",
                    dimensions=get_sentence_embedding_dimensions(),
                )
            ],
            guidelines="Please review the text and provide or correct the label where needed.",
        )

        dataframe["text_length"] = dataframe["text"].apply(len)
        dataframe["text_embeddings"] = get_embeddings(dataframe["text"])

        progress(0.5, desc="Creating dataset")
        rg_dataset = client.datasets(name=repo_name, workspace=hf_user)
        if rg_dataset is None:
            rg_dataset = rg.Dataset(
                name=repo_name,
                workspace=hf_user,
                settings=settings,
                client=client,
            )
            rg_dataset = rg_dataset.create()
        progress(0.7, desc="Pushing dataset to Argilla")
        hf_dataset = Dataset.from_pandas(dataframe)
        records = [
            rg.Record(
                fields={
                    "text": sample["text"],
                },
                metadata={"text_length": sample["text_length"]},
                vectors={"text_embeddings": sample["text_embeddings"]},
                suggestions=(
                    [
                        rg.Suggestion(
                            question_name="label" if num_labels == 1 else "labels",
                            value=(
                                sample["label"] if num_labels == 1 else sample["labels"]
                            ),
                        )
                    ]
                    if (
                        (num_labels == 1 and sample["label"] in labels)
                        or (
                            num_labels > 1
                            and all(label in labels for label in sample["labels"])
                        )
                    )
                    else []
                ),
            )
            for sample in hf_dataset
        ]
        rg_dataset.records.log(records=records)
        progress(1.0, desc="Dataset pushed to Argilla")
    except Exception as e:
        raise gr.Error(f"Error pushing dataset to Argilla: {e}")
    return ""


def update_suggested_labels(system_prompt):
    new_labels = re.findall(r"'(\b[\w-]+\b)'", system_prompt)
    if not new_labels:
        return gr.Warning(
            "No labels found in the system prompt. Please add labels manually."
        )
    return gr.update(choices=new_labels, value=new_labels)


def validate_input_labels(labels):
    if not labels or len(labels) < 2:
        raise gr.Error(
            f"Please select at least 2 labels to classify your text. You selected {len(labels) if labels else 0}."
        )
    return labels


def update_max_num_labels(labels):
    return gr.update(maximum=len(labels) if labels else 1)


with gr.Blocks() as app:
    gr.Markdown("## Describe the dataset you want")
    gr.HTML("<hr>")
    with gr.Row():
        with gr.Column(scale=1):
            dataset_description = gr.Textbox(
                label="Dataset description",
                placeholder="Give a precise description of your desired dataset.",
            )
            examples = gr.Examples(
                examples=DEFAULT_DATASET_DESCRIPTIONS,
                inputs=[dataset_description],
                cache_examples=False,
                label="Example descriptions",
            )
            system_prompt = gr.Textbox(
                label="System prompt",
                placeholder="You are a helpful assistant.",
                visible=False,
            )
            load_btn = gr.Button("Load Dataset")
        with gr.Column(scale=3):
            pass

    gr.Markdown("## Configure your task")
    gr.HTML("<hr>")
    with gr.Row():
        with gr.Column(scale=1):
            difficulty = gr.Dropdown(
                choices=[
                    ("High School", "high school"),
                    ("College", "college"),
                    ("PhD", "PhD"),
                    ("Mixed", "mixed"),
                ],
                value="mixed",
                label="Difficulty",
                info="Select the comprehension level for the text. Ensure it matches the task context.",
                interactive=True,
            )
            clarity = gr.Dropdown(
                choices=[
                    ("Clear", "clear"),
                    (
                        "Understandable",
                        "understandable with some effort",
                    ),
                    ("Ambiguous", "ambiguous"),
                    ("Mixed", "mixed"),
                ],
                value="mixed",
                label="Clarity",
                info="Set how easily the correct label or labels can be identified.",
                interactive=True,
            )
            labels = gr.Dropdown(
                choices=[],
                allow_custom_value=True,
                interactive=True,
                label="Labels",
                multiselect=True,
                info="Add the labels to classify the text.",
            )
            num_labels = gr.Number(
                label="Number of labels per text",
                value=1,
                minimum=1,
                maximum=10,
                info="Select 1 for single-label and >1 for multi-label.",
                interactive=True,
            )
            btn_apply_to_sample_dataset = gr.Button("Refresh dataset")
        with gr.Column(scale=3):
            dataframe = gr.Dataframe()

    gr.Markdown("## Generate your dataset")
    gr.HTML("<hr>")
    with gr.Row():
        with gr.Column(scale=1):
            org_name = get_org_dropdown()
            repo_name = gr.Textbox(
                label="Repo name",
                placeholder="dataset_name",
                value=f"my-distiset-{str(uuid.uuid4())[:8]}",
                interactive=True,
            )
            n_rows = gr.Number(
                label="Number of rows",
                value=10,
                interactive=True,
                scale=1,
            )
            private = gr.Checkbox(
                label="Private dataset",
                value=False,
                interactive=True,
                scale=1,
            )
            btn_push_to_hub = gr.Button("Push to Hub", variant="primary", scale=2)
        with gr.Column(scale=3):
            success_message = gr.Markdown(visible=True)

    pipeline_code = get_pipeline_code_ui(
        generate_pipeline_code(
            system_prompt.value,
            difficulty=difficulty.value,
            clarity=clarity.value,
            labels=labels.value,
            num_labels=num_labels.value,
            num_rows=n_rows.value,
        )
    )

    gr.on(
        triggers=[load_btn.click, btn_apply_to_sample_dataset.click],
        fn=generate_system_prompt,
        inputs=[dataset_description],
        outputs=[system_prompt, dataframe],
        show_progress=True,
    ).then(
        fn=generate_sample_dataset,
        inputs=[system_prompt],
        outputs=[dataframe],
        show_progress=True,
    ).then(
        fn=update_suggested_labels,
        inputs=[system_prompt],
        outputs=labels,
    ).then(
        fn=update_max_num_labels,
        inputs=[labels],
        outputs=[num_labels],
    )

    btn_push_to_hub.click(
        fn=validate_argilla_user_workspace_dataset,
        inputs=[repo_name],
        outputs=[success_message],
        show_progress=True,
    ).then(
        fn=validate_push_to_hub,
        inputs=[org_name, repo_name],
        outputs=[success_message],
        show_progress=True,
    ).success(
        fn=hide_success_message,
        outputs=[success_message],
        show_progress=True,
    ).success(
        fn=push_dataset_to_argilla,
        inputs=[
            org_name,
            repo_name,
            system_prompt,
            difficulty,
            clarity,
            num_labels,
            n_rows,
            labels,
            private,
        ],
        outputs=[success_message],
        show_progress=True,
    ).success(
        fn=show_success_message_hub,
        inputs=[org_name, repo_name],
        outputs=[success_message],
    )

    app.load(fn=get_org_dropdown, outputs=[org_name])
