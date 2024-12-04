import json
import uuid
from typing import List, Union

import argilla as rg
import gradio as gr
import pandas as pd
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from distilabel.distiset import Distiset
from huggingface_hub import HfApi

from src.synthetic_dataset_generator.apps.base import (
    hide_success_message,
    show_success_message,
    validate_argilla_user_workspace_dataset,
    validate_push_to_hub,
)
from src.synthetic_dataset_generator.pipelines.embeddings import (
    get_embeddings,
    get_sentence_embedding_dimensions,
)
from src.synthetic_dataset_generator.pipelines.textcat import (
    DEFAULT_DATASET_DESCRIPTIONS,
    generate_pipeline_code,
    get_labeller_generator,
    get_prompt_generator,
    get_textcat_generator,
)
from src.synthetic_dataset_generator.utils import (
    get_argilla_client,
    get_org_dropdown,
    get_preprocess_labels,
    swap_visibility,
)
from synthetic_dataset_generator.constants import DEFAULT_BATCH_SIZE


def _get_dataframe():
    return gr.Dataframe(
        headers=["labels", "text"], wrap=True, height=500, interactive=False
    )


def generate_system_prompt(dataset_description, temperature, progress=gr.Progress()):
    progress(0.0, desc="Generating text classification task")
    progress(0.3, desc="Initializing text generation")
    generate_description = get_prompt_generator(temperature)
    progress(0.7, desc="Generating text classification task")
    result = next(
        generate_description.process(
            [
                {
                    "instruction": dataset_description,
                }
            ]
        )
    )[0]["generation"]
    progress(1.0, desc="Text classification task generated")
    data = json.loads(result)
    system_prompt = data["classification_task"]
    labels = data["labels"]
    return system_prompt, labels


def generate_sample_dataset(
    system_prompt, difficulty, clarity, labels, num_labels, progress=gr.Progress()
):
    dataframe = generate_dataset(
        system_prompt=system_prompt,
        difficulty=difficulty,
        clarity=clarity,
        labels=labels,
        num_labels=num_labels,
        num_rows=10,
        progress=progress,
        is_sample=True,
    )
    return dataframe


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
    progress(0.0, desc="(1/2) Generating text classification data")
    labels = get_preprocess_labels(labels)
    textcat_generator = get_textcat_generator(
        difficulty=difficulty, clarity=clarity, is_sample=is_sample
    )
    labeller_generator = get_labeller_generator(
        system_prompt=f"{system_prompt} {', '.join(labels)}",
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
            2 * 0.5 * n_processed / num_rows,
            total=total_steps,
            desc="(1/2) Generating text classification data",
        )
        remaining_rows = num_rows - n_processed
        batch_size = min(batch_size, remaining_rows)
        inputs = [
            {"task": f"{system_prompt} {', '.join(labels)}"} for _ in range(batch_size)
        ]
        batch = list(textcat_generator.process(inputs=inputs))
        textcat_results.extend(batch[0])
        n_processed += batch_size
    for result in textcat_results:
        result["text"] = result["input_text"]

    # label text classification data
    progress(2 * 0.5, desc="(1/2) Generating text classification data")
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
    for result in labeller_results:
        record = {key: result[key] for key in ["labels", "text"] if key in result}
        distiset_results.append(record)

    dataframe = pd.DataFrame(distiset_results)
    if num_labels == 1:
        dataframe = dataframe.rename(columns={"labels": "label"})
        dataframe["label"] = dataframe["label"].apply(
            lambda x: x.lower().strip() if x.lower().strip() in labels else None
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


def push_dataset(
    org_name: str,
    repo_name: str,
    system_prompt: str,
    difficulty: str,
    clarity: str,
    num_labels: int = 1,
    num_rows: int = 10,
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
        num_rows=num_rows,
    )
    push_dataset_to_hub(
        dataframe, org_name, repo_name, num_labels, labels, oauth_token, private
    )

    dataframe = dataframe[
        (dataframe["text"].str.strip() != "") & (dataframe["text"].notna())
    ]
    try:
        progress(0.1, desc="Setting up user and workspace")
        hf_user = HfApi().whoami(token=oauth_token.token)["name"]
        client = get_argilla_client()
        if client is None:
            return ""
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
        dataframe["text_embeddings"] = get_embeddings(dataframe["text"].to_list())

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


def validate_input_labels(labels):
    if not labels or len(labels) < 2:
        raise gr.Error(
            f"Please select at least 2 labels to classify your text. You selected {len(labels) if labels else 0}."
        )
    return labels


def update_max_num_labels(labels):
    return gr.update(maximum=len(labels) if labels else 1)


def show_pipeline_code_visibility():
    return {pipeline_code_ui: gr.Accordion(visible=True)}


def hide_pipeline_code_visibility():
    return {pipeline_code_ui: gr.Accordion(visible=False)}


######################
# Gradio UI
######################


with gr.Blocks() as app:
    with gr.Column() as main_ui:
        gr.Markdown("## 1. Describe the dataset you want")
        with gr.Row():
            with gr.Column(scale=2):
                dataset_description = gr.Textbox(
                    label="Dataset description",
                    placeholder="Give a precise description of your desired dataset.",
                )
                with gr.Row():
                    clear_btn_part = gr.Button(
                        "Clear",
                        variant="secondary",
                    )
                    load_btn = gr.Button(
                        "Create",
                        variant="primary",
                    )
            with gr.Column(scale=3):
                examples = gr.Examples(
                    examples=DEFAULT_DATASET_DESCRIPTIONS,
                    inputs=[dataset_description],
                    cache_examples=False,
                    label="Examples",
                )

        gr.HTML("<hr>")
        gr.Markdown("## 2. Configure your dataset")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                system_prompt = gr.Textbox(
                    label="System prompt",
                    placeholder="You are a helpful assistant.",
                    visible=True,
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
                with gr.Row():
                    clear_btn_full = gr.Button("Clear", variant="secondary")
                    btn_apply_to_sample_dataset = gr.Button("Save", variant="primary")
            with gr.Column(scale=3):
                dataframe = _get_dataframe()

        gr.HTML("<hr>")
        gr.Markdown("## 3. Generate your dataset")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                org_name = get_org_dropdown()
                repo_name = gr.Textbox(
                    label="Repo name",
                    placeholder="dataset_name",
                    value=f"my-distiset-{str(uuid.uuid4())[:8]}",
                    interactive=True,
                )
                num_rows = gr.Number(
                    label="Number of rows",
                    value=10,
                    interactive=True,
                    scale=1,
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=1,
                    value=0.8,
                    step=0.1,
                    interactive=True,
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
                with gr.Accordion(
                    "Customize your pipeline with distilabel",
                    open=False,
                    visible=False,
                ) as pipeline_code_ui:
                    code = generate_pipeline_code(
                        system_prompt.value,
                        difficulty=difficulty.value,
                        clarity=clarity.value,
                        labels=labels.value,
                        num_labels=num_labels.value,
                        num_rows=num_rows.value,
                    )
                    pipeline_code = gr.Code(
                        value=code,
                        language="python",
                        label="Distilabel Pipeline Code",
                    )

    load_btn.click(
        fn=generate_system_prompt,
        inputs=[dataset_description, temperature],
        outputs=[system_prompt, labels],
        show_progress=True,
    ).then(
        fn=generate_sample_dataset,
        inputs=[system_prompt, difficulty, clarity, labels, num_labels],
        outputs=[dataframe],
        show_progress=True,
    ).then(
        fn=update_max_num_labels,
        inputs=[labels],
        outputs=[num_labels],
    )

    labels.input(
        fn=update_max_num_labels,
        inputs=[labels],
        outputs=[num_labels],
    )

    btn_apply_to_sample_dataset.click(
        fn=generate_sample_dataset,
        inputs=[system_prompt, difficulty, clarity, labels, num_labels],
        outputs=[dataframe],
        show_progress=True,
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
        fn=hide_pipeline_code_visibility,
        inputs=[],
        outputs=[pipeline_code_ui],
    ).success(
        fn=push_dataset,
        inputs=[
            org_name,
            repo_name,
            system_prompt,
            difficulty,
            clarity,
            num_labels,
            num_rows,
            labels,
            private,
        ],
        outputs=[success_message],
        show_progress=True,
    ).success(
        fn=show_success_message,
        inputs=[org_name, repo_name],
        outputs=[success_message],
    ).success(
        fn=generate_pipeline_code,
        inputs=[
            system_prompt,
            difficulty,
            clarity,
            labels,
            num_labels,
            num_rows,
        ],
        outputs=[pipeline_code],
    ).success(
        fn=show_pipeline_code_visibility,
        inputs=[],
        outputs=[pipeline_code_ui],
    )

    gr.on(
        triggers=[clear_btn_part.click, clear_btn_full.click],
        fn=lambda _: (
            "",
            "",
            [],
            _get_dataframe(),
        ),
        inputs=[dataframe],
        outputs=[dataset_description, system_prompt, labels, dataframe],
    )

    app.load(fn=swap_visibility, outputs=main_ui)
    app.load(fn=get_org_dropdown, outputs=[org_name])
