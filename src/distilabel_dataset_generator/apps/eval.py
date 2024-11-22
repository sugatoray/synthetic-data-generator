import json

import gradio as gr
import pandas as pd
from datasets import load_dataset
from gradio_huggingfacehub_search import HuggingfaceHubSearch

from src.distilabel_dataset_generator.utils import get_org_dropdown


def get_iframe(hub_repo_id) -> str:
    if not hub_repo_id:
        raise gr.Error("Hub repo id is required")
    url = f"https://huggingface.co/datasets/{hub_repo_id}/embed/viewer"
    iframe = f"""
    <iframe
  src="{url}"
  frameborder="0"
  width="100%"
  height="600px"
></iframe>
"""
    return iframe


def get_valid_columns(df: pd.DataFrame):
    valid_columns = []
    for col in df.columns:
        sample_val = df[col].iloc[0]
        if isinstance(sample_val, str) or (
            isinstance(sample_val, list)
            and all(isinstance(item, dict) for item in sample_val)
        ):
            valid_columns.append(col)
    return valid_columns


def load_dataset_from_hub(hub_repo_id: str, n_rows: int = 10):
    gr.Info(message="Loading dataset ...")
    if not hub_repo_id:
        raise gr.Error("Hub repo id is required")
    ds_dict = load_dataset(hub_repo_id)
    splits = list(ds_dict.keys())
    ds = ds_dict[splits[0]]
    if n_rows:
        ds = ds.select(range(n_rows))
    df = ds.to_pandas()
    # Get columns that contain either strings or lists of dictionaries
    valid_columns = get_valid_columns(df)
    return (
        df,
        gr.Dropdown(choices=valid_columns, label="Instruction Column"),
        gr.Dropdown(choices=valid_columns, label="Instruction Column"),
        gr.Dropdown(choices=valid_columns, label="Response Column"),
    )


def define_evaluation_aspects(task_type: str):
    if task_type == "instruction":
        return gr.Dropdown(
            value=["overall-rating"],
            choices=["complexity", "quality"],
            label="Evaluation Aspects",
            multiselect=True,
            interactive=True,
        )
    elif task_type == "instruction-response":
        return gr.Dropdown(
            value=["overall-rating"],
            choices=["helpfulness", "truthfulness", "overall-rating", "honesty"],
            label="Evaluation Aspects",
            multiselect=True,
            interactive=True,
        )
    else:
        return gr.Dropdown(interactive=False)


def evaluate_instruction(df: pd.DataFrame, aspects: list[str], instruction_column: str):
    pass


def evaluate_instruction_response(
    df: pd.DataFrame, aspects: list[str], instruction_column: str, response_column: str
):
    pass


def evaluate_custom(
    df: pd.DataFrame, aspects: list[str], prompt_template: str, structured_output: dict
):
    pass


def _apply_to_dataset(
    df: pd.DataFrame,
    eval_type: str,
    aspects_instruction: list[str],
    instruction_column: str,
    aspects_instruction_response: list[str],
    instruction_column_response: str,
    response_column_response: str,
    aspects_custom: list[str],
    prompt_template: str,
    structured_output: dict,
):
    if eval_type == "instruction":
        df = evaluate_instruction(df, aspects_instruction, instruction_column)
    elif eval_type == "instruction-response":
        df = evaluate_instruction_response(
            df,
            aspects_instruction_response,
            instruction_column_response,
            response_column_response,
        )
    elif eval_type == "custom":
        df = evaluate_custom(df, aspects_custom, prompt_template, structured_output)
    return df


def apply_to_sample_dataset(
    repo_id: str,
    eval_type: str,
    aspects_instruction: list[str],
    aspects_instruction_response: list[str],
    aspects_custom: list[str],
    instruction_instruction: str,
    instruction_instruction_response: str,
    response_instruction_response: str,
    prompt_template: str,
    structured_output: dict,
):
    df, _, _, _ = load_dataset_from_hub(repo_id, n_rows=10)
    df = _apply_to_dataset(
        df,
        eval_type,
        aspects_instruction,
        instruction_instruction,
        aspects_instruction_response,
        instruction_instruction_response,
        response_instruction_response,
        aspects_custom,
        prompt_template,
        structured_output,
    )
    return df


def push_to_hub(
    org_name: str,
    repo_name: str,
    private: bool,
    n_rows: int,
    original_repo_id: str,
    eval_type: str,
    aspects_instruction: list[str],
    aspects_instruction_response: list[str],
    aspects_custom: list[str],
    instruction_instruction: str,
    instruction_instruction_response: str,
    response_instruction_response: str,
    prompt_template: str,
    structured_output: dict,
):
    df, _, _, _ = load_dataset_from_hub(original_repo_id, n_rows=n_rows)
    df = _apply_to_dataset(
        df,
        eval_type,
        aspects_instruction,
        instruction_instruction,
        aspects_instruction_response,
        instruction_instruction_response,
        response_instruction_response,
        aspects_custom,
        prompt_template,
        structured_output,
    )
    new_repo_id = f"{org_name}/{repo_name}"
    print(df)


with gr.Blocks() as app:
    gr.Markdown("## Select your input dataset")
    gr.HTML("<hr>")
    with gr.Row():
        with gr.Column(scale=1):
            search_in = HuggingfaceHubSearch(
                label="Search",
                placeholder="Search for a Dataset",
                search_type="dataset",
                sumbit_on_select=True,
            )
            load_btn = gr.Button("Load Dataset")
        with gr.Column(scale=3):
            search_out = gr.HTML(label="Dataset Preview")

    gr.Markdown("## Configure your task")
    gr.HTML("<hr>")
    with gr.Row():
        with gr.Column(scale=1):
            eval_type = gr.Dropdown(
                label="Evaluation Type",
                choices=["instruction", "instruction-response", "custom-template"],
                visible=False,
            )
            with gr.Tab("instruction") as tab_instruction:
                aspects_instruction = define_evaluation_aspects("instruction")
                instruction_instruction = gr.Dropdown(
                    label="Instruction Column", interactive=True
                )
                tab_instruction.select(
                    lambda: "instruction",
                    inputs=[],
                    outputs=[eval_type],
                )
            with gr.Tab("instruction-response") as tab_instruction_response:
                aspects_instruction_response = define_evaluation_aspects(
                    "instruction-response"
                )
                instruction_instruction_response = gr.Dropdown(
                    label="Instruction Column", interactive=True
                )
                response_instruction_response = gr.Dropdown(
                    label="Response Column", interactive=True
                )
                tab_instruction_response.select(
                    lambda: "instruction-response",
                    inputs=[],
                    outputs=[eval_type],
                )
            with gr.Tab("custom") as tab_custom:
                aspects_custom = define_evaluation_aspects("custom")
                prompt_template = gr.Code(
                    label="Prompt Template",
                    value="{{column_1}} based on {{column_2}}",
                    language="markdown",
                    interactive=True,
                )
                structured_output = gr.Code(
                    label="Structured Output",
                    value=json.dumps({"eval_aspect": "str"}),
                    language="json",
                    interactive=True,
                )
                tab_custom.select(
                    lambda: "custom-template",
                    inputs=[],
                    outputs=[eval_type],
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
                value="my-distiset",
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
            success_message = gr.Markdown(visible=False)

    search_in.submit(get_iframe, inputs=search_in, outputs=search_out)
    load_btn.click(
        load_dataset_from_hub,
        inputs=[search_in],
        outputs=[
            dataframe,
            instruction_instruction,
            instruction_instruction_response,
            response_instruction_response,
        ],
    )
    btn_apply_to_sample_dataset.click(
        apply_to_sample_dataset,
        inputs=[
            search_in,
            eval_type,
            aspects_instruction,
            aspects_instruction_response,
            aspects_custom,
            instruction_instruction,
            instruction_instruction_response,
            response_instruction_response,
            prompt_template,
            structured_output,
        ],
        outputs=dataframe,
    )
    btn_push_to_hub.click(
        push_to_hub,
        inputs=[
            org_name,
            repo_name,
            private,
            n_rows,
            search_in,
            eval_type,
            aspects_instruction,
            aspects_instruction_response,
            aspects_custom,
            instruction_instruction,
            instruction_instruction_response,
            response_instruction_response,
            prompt_template,
            structured_output,
        ],
        outputs=success_message,
    )
    app.load(fn=get_org_dropdown, outputs=[org_name])
