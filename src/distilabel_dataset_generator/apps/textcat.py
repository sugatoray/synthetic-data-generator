import re
from typing import List, Union

import argilla as rg
import gradio as gr
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi

from src.distilabel_dataset_generator.apps.base import (
    get_argilla_client,
    get_main_ui,
    get_pipeline_code_ui,
    hide_success_message,
    push_pipeline_code_to_hub,
    show_success_message_argilla,
    show_success_message_hub,
    validate_argilla_user_workspace_dataset,
)
from src.distilabel_dataset_generator.apps.base import (
    push_dataset_to_hub as push_to_hub_base,
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
    DEFAULT_DATASETS,
    DEFAULT_SYSTEM_PROMPTS,
    PROMPT_CREATION_PROMPT,
    generate_pipeline_code,
    get_labeller_generator,
    get_prompt_generator,
    get_textcat_generator,
)
from src.distilabel_dataset_generator.utils import get_preprocess_labels

TASK = "text_classification"


def push_dataset_to_hub(
    dataframe: pd.DataFrame,
    private: bool = True,
    org_name: str = None,
    repo_name: str = None,
    oauth_token: Union[gr.OAuthToken, None] = None,
    progress=gr.Progress(),
    labels: List[str] = None,
    num_labels: int = 1,
):
    original_dataframe = dataframe.copy(deep=True)
    labels = get_preprocess_labels(labels)
    try:
        push_to_hub_base(
            dataframe,
            private,
            org_name,
            repo_name,
            oauth_token,
            progress,
            labels,
            num_labels,
            task=TASK,
        )
    except Exception as e:
        raise gr.Error(f"Error pushing dataset to the Hub: {e}")
    return original_dataframe


def push_dataset_to_argilla(
    dataframe: pd.DataFrame,
    dataset_name: str,
    oauth_token: Union[gr.OAuthToken, None] = None,
    progress=gr.Progress(),
    num_labels: int = 1,
    labels: List[str] = None,
) -> pd.DataFrame:
    original_dataframe = dataframe.copy(deep=True)
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
        rg_dataset = client.datasets(name=dataset_name, workspace=hf_user)
        if rg_dataset is None:
            rg_dataset = rg.Dataset(
                name=dataset_name,
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
    return original_dataframe


def generate_system_prompt(dataset_description, progress=gr.Progress()):
    progress(0.0, desc="Generating text classification task")
    if dataset_description in DEFAULT_DATASET_DESCRIPTIONS:
        index = DEFAULT_DATASET_DESCRIPTIONS.index(dataset_description)
        if index < len(DEFAULT_SYSTEM_PROMPTS):
            return DEFAULT_SYSTEM_PROMPTS[index]

    progress(0.3, desc="Initializing text generation")
    generate_description = get_prompt_generator()
    progress(0.7, desc="Generating text classification task")
    result = next(
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
    return result


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
            0.5 * n_processed / num_rows,
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
    progress(0.5, desc="(1/2) Generating text classification data")
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


(
    app,
    main_ui,
    custom_input_ui,
    dataset_description,
    examples,
    btn_generate_system_prompt,
    system_prompt,
    sample_dataset,
    btn_generate_sample_dataset,
    dataset_name,
    add_to_existing_dataset,
    btn_generate_full_dataset_argilla,
    btn_generate_and_push_to_argilla,
    btn_push_to_argilla,
    org_name,
    repo_name,
    private,
    btn_generate_full_dataset,
    btn_generate_and_push_to_hub,
    btn_push_to_hub,
    final_dataset,
    success_message,
) = get_main_ui(
    default_dataset_descriptions=DEFAULT_DATASET_DESCRIPTIONS,
    default_system_prompts=DEFAULT_SYSTEM_PROMPTS,
    default_datasets=DEFAULT_DATASETS,
    fn_generate_system_prompt=generate_system_prompt,
    fn_generate_dataset=generate_dataset,
    task=TASK,
)

with app:
    with main_ui:
        with custom_input_ui:
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
            )
            with gr.Column():
                labels = gr.Dropdown(
                    choices=[],
                    allow_custom_value=True,
                    interactive=True,
                    label="Labels",
                    multiselect=True,
                    info="Add the labels to classify the text.",
                )
                with gr.Blocks():
                    btn_suggested_labels = gr.Button(
                        value="Add suggested labels",
                        size="sm",
                    )
            num_labels = gr.Number(
                label="Number of labels per text",
                value=1,
                minimum=1,
                maximum=10,
                info="Select 1 for single-label and >1 for multi-label.",
            )
            num_rows = gr.Number(
                label="Number of rows",
                value=10,
                minimum=1,
                maximum=500,
                info="Select the number of rows in the dataset. More rows will take more time.",
            )

        pipeline_code = get_pipeline_code_ui(
            generate_pipeline_code(
                system_prompt.value,
                difficulty=difficulty.value,
                clarity=clarity.value,
                labels=labels.value,
                num_labels=num_labels.value,
                num_rows=num_rows.value,
            )
        )

    # define app triggers
    btn_suggested_labels.click(
        fn=update_suggested_labels,
        inputs=[system_prompt],
        outputs=labels,
    ).then(
        fn=update_max_num_labels,
        inputs=[labels],
        outputs=[num_labels],
    )

    gr.on(
        triggers=[
            btn_generate_full_dataset.click,
            btn_generate_full_dataset_argilla.click,
        ],
        fn=hide_success_message,
        outputs=[success_message],
    ).then(
        fn=validate_input_labels,
        inputs=[labels],
        outputs=[labels],
    ).success(
        fn=generate_dataset,
        inputs=[system_prompt, difficulty, clarity, labels, num_labels, num_rows],
        outputs=[final_dataset],
        show_progress=True,
    )

    btn_generate_and_push_to_argilla.click(
        fn=validate_argilla_user_workspace_dataset,
        inputs=[dataset_name, final_dataset, add_to_existing_dataset],
        outputs=[final_dataset],
        show_progress=True,
    ).success(
        fn=hide_success_message,
        outputs=[success_message],
    ).success(
        fn=generate_dataset,
        inputs=[system_prompt, difficulty, clarity, labels, num_labels, num_rows],
        outputs=[final_dataset],
        show_progress=True,
    ).success(
        fn=push_dataset_to_argilla,
        inputs=[final_dataset, dataset_name, num_labels, labels],
        outputs=[final_dataset],
        show_progress=True,
    ).success(
        fn=show_success_message_argilla,
        inputs=[],
        outputs=[success_message],
    )

    btn_generate_and_push_to_hub.click(
        fn=hide_success_message,
        outputs=[success_message],
    ).then(
        fn=generate_dataset,
        inputs=[system_prompt, difficulty, clarity, labels, num_labels, num_rows],
        outputs=[final_dataset],
        show_progress=True,
    ).then(
        fn=push_dataset_to_hub,
        inputs=[final_dataset, private, org_name, repo_name, labels, num_labels],
        outputs=[final_dataset],
        show_progress=True,
    ).then(
        fn=push_pipeline_code_to_hub,
        inputs=[pipeline_code, org_name, repo_name],
        outputs=[],
        show_progress=True,
    ).success(
        fn=show_success_message_hub,
        inputs=[org_name, repo_name],
        outputs=[success_message],
    )

    btn_push_to_hub.click(
        fn=hide_success_message,
        outputs=[success_message],
    ).then(
        fn=push_dataset_to_hub,
        inputs=[final_dataset, private, org_name, repo_name, labels, num_labels],
        outputs=[final_dataset],
        show_progress=True,
    ).then(
        fn=push_pipeline_code_to_hub,
        inputs=[pipeline_code, org_name, repo_name],
        outputs=[],
        show_progress=True,
    ).success(
        fn=show_success_message_hub,
        inputs=[org_name, repo_name],
        outputs=[success_message],
    )

    btn_push_to_argilla.click(
        fn=hide_success_message,
        outputs=[success_message],
    ).success(
        fn=validate_argilla_user_workspace_dataset,
        inputs=[dataset_name, final_dataset, add_to_existing_dataset],
        outputs=[final_dataset],
        show_progress=True,
    ).success(
        fn=push_dataset_to_argilla,
        inputs=[final_dataset, dataset_name, num_labels, labels],
        outputs=[final_dataset],
        show_progress=True,
    ).success(
        fn=show_success_message_argilla,
        inputs=[],
        outputs=[success_message],
    )

    system_prompt.change(
        fn=generate_pipeline_code,
        inputs=[system_prompt, difficulty, clarity, labels, num_labels, num_rows],
        outputs=[pipeline_code],
    )
    difficulty.change(
        fn=generate_pipeline_code,
        inputs=[system_prompt, difficulty, clarity, labels, num_labels, num_rows],
        outputs=[pipeline_code],
    )
    clarity.change(
        fn=generate_pipeline_code,
        inputs=[system_prompt, difficulty, clarity, labels, num_labels, num_rows],
        outputs=[pipeline_code],
    )
    labels.change(
        fn=generate_pipeline_code,
        inputs=[system_prompt, difficulty, clarity, labels, num_labels, num_rows],
        outputs=[pipeline_code],
    ).then(
        fn=update_max_num_labels,
        inputs=[labels],
        outputs=[num_labels],
    )
    num_labels.change(
        fn=generate_pipeline_code,
        inputs=[system_prompt, difficulty, clarity, labels, num_labels, num_rows],
        outputs=[pipeline_code],
    )
    num_rows.change(
        fn=generate_pipeline_code,
        inputs=[system_prompt, difficulty, clarity, labels, num_labels, num_rows],
        outputs=[pipeline_code],
    )
