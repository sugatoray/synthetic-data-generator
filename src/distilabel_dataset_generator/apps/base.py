import io
import uuid
from typing import Any, Callable, List, Tuple, Union

import argilla as rg
import gradio as gr
import pandas as pd
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from distilabel.distiset import Distiset
from gradio import OAuthToken
from huggingface_hub import HfApi, upload_file

from distilabel_dataset_generator.utils import (
    _LOGGED_OUT_CSS,
    get_argilla_client,
    get_login_button,
    list_orgs,
    swap_visibility,
)

TEXTCAT_TASK = "text_classification"
SFT_TASK = "supervised_fine_tuning"


def get_main_ui(
    default_dataset_descriptions: List[str],
    default_system_prompts: List[str],
    default_datasets: List[pd.DataFrame],
    fn_generate_system_prompt: Callable,
    fn_generate_dataset: Callable,
    task: str,
):
    def fn_generate_sample_dataset(system_prompt, progress=gr.Progress()):
        if system_prompt in default_system_prompts:
            index = default_system_prompts.index(system_prompt)
            if index < len(default_datasets):
                return default_datasets[index]
        if task == TEXTCAT_TASK:
            result = fn_generate_dataset(
                system_prompt=system_prompt,
                difficulty="high school",
                clarity="clear",
                labels=[],
                num_labels=1,
                num_rows=1,
                progress=progress,
                is_sample=True,
            )
        else:
            result = fn_generate_dataset(
                system_prompt=system_prompt,
                num_turns=1,
                num_rows=1,
                progress=progress,
                is_sample=True,
            )
        return result

    with gr.Blocks(
        title="🧬 Synthetic Data Generator",
        head="🧬 Synthetic Data Generator",
        css=_LOGGED_OUT_CSS,
    ) as app:
        with gr.Row():
            gr.HTML(
                """<details style='display: inline-block;'><summary><h2 style='display: inline;'>How does it work?</h2></summary><img src='https://huggingface.co/spaces/argilla/synthetic-data-generator/resolve/main/assets/flow.png' width='100%' style='margin: 0 auto; display: block;'></details>"""
            )
        with gr.Row():
            gr.Markdown(
                "Want to run this locally or with other LLMs? Take a look at the FAQ tab. distilabel Synthetic Data Generator is free, we use the authentication token to push the dataset to the Hugging Face Hub and not for data generation."
            )
        with gr.Row():
            gr.Column()
            get_login_button()
            gr.Column()

        gr.Markdown("## Iterate on a sample dataset")
        with gr.Column() as main_ui:
            (
                dataset_description,
                examples,
                btn_generate_system_prompt,
                system_prompt,
                sample_dataset,
                btn_generate_sample_dataset,
            ) = get_iterate_on_sample_dataset_ui(
                default_dataset_descriptions=default_dataset_descriptions,
                default_system_prompts=default_system_prompts,
                default_datasets=default_datasets,
                task=task,
            )
            gr.Markdown("## Generate full dataset")
            gr.Markdown(
                "Once you're satisfied with the sample, generate a larger dataset and push it to Argilla or the Hugging Face Hub."
            )
            with gr.Row(variant="panel") as custom_input_ui:
                pass

            (
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
            ) = get_push_to_ui(default_datasets)

        sample_dataset.change(
            fn=lambda x: x,
            inputs=[sample_dataset],
            outputs=[final_dataset],
        )

        btn_generate_system_prompt.click(
            fn=fn_generate_system_prompt,
            inputs=[dataset_description],
            outputs=[system_prompt],
            show_progress=True,
        ).then(
            fn=fn_generate_sample_dataset,
            inputs=[system_prompt],
            outputs=[sample_dataset],
            show_progress=True,
        )

        btn_generate_sample_dataset.click(
            fn=fn_generate_sample_dataset,
            inputs=[system_prompt],
            outputs=[sample_dataset],
            show_progress=True,
        )

        app.load(fn=swap_visibility, outputs=main_ui)
        app.load(get_org_dropdown, outputs=[org_name])

    return (
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
    )


def validate_argilla_user_workspace_dataset(
    dataset_name: str,
    add_to_existing_dataset: bool = True,
    oauth_token: Union[OAuthToken, None] = None,
    progress=gr.Progress(),
) -> str:
    progress(0, desc="Validating dataset configuration")
    hf_user = HfApi().whoami(token=oauth_token.token)["name"]
    client = get_argilla_client()
    if dataset_name is None or dataset_name == "":
        raise gr.Error("Dataset name is required")
    # Create user if it doesn't exist
    rg_user = client.users(username=hf_user)
    if rg_user is None:
        rg_user = client.users.add(
            rg.User(username=hf_user, role="admin", password=str(uuid.uuid4()))
        )
    # Create workspace if it doesn't exist
    workspace = client.workspaces(name=hf_user)
    if workspace is None:
        workspace = client.workspaces.add(rg.Workspace(name=hf_user))
        workspace.add_user(hf_user)
    # Check if dataset exists
    dataset = client.datasets(name=dataset_name, workspace=hf_user)
    if dataset and not add_to_existing_dataset:
        raise gr.Error(f"Dataset {dataset_name} already exists")
    return ""


def get_org_dropdown(oauth_token: Union[OAuthToken, None]):
    orgs = list_orgs(oauth_token)
    return gr.Dropdown(
        label="Organization",
        choices=orgs,
        value=orgs[0] if orgs else None,
        allow_custom_value=True,
    )


def get_push_to_ui(default_datasets):
    with gr.Column() as push_to_ui:
        (
            dataset_name,
            add_to_existing_dataset,
            btn_generate_full_dataset_argilla,
            btn_generate_and_push_to_argilla,
            btn_push_to_argilla,
        ) = get_argilla_tab()
        (
            org_name,
            repo_name,
            private,
            btn_generate_full_dataset,
            btn_generate_and_push_to_hub,
            btn_push_to_hub,
        ) = get_hf_tab()
        final_dataset = get_final_dataset_row(default_datasets)
        success_message = get_success_message_row()
    return (
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
    )


def get_iterate_on_sample_dataset_ui(
    default_dataset_descriptions: List[str],
    default_system_prompts: List[str],
    default_datasets: List[pd.DataFrame],
    task: str,
):
    with gr.Column():
        dataset_description = gr.TextArea(
            label="Give a precise description of your desired application. Check the examples for inspiration.",
            value=default_dataset_descriptions[0],
            lines=2,
        )
        examples = gr.Examples(
            elem_id="system_prompt_examples",
            examples=[[example] for example in default_dataset_descriptions],
            inputs=[dataset_description],
        )
        with gr.Row():
            gr.Column(scale=1)
            btn_generate_system_prompt = gr.Button(
                value="Generate system prompt and sample dataset", variant="primary"
            )
            gr.Column(scale=1)

        system_prompt = gr.TextArea(
            label="System prompt for dataset generation. You can tune it and regenerate the sample.",
            value=default_system_prompts[0],
            lines=2 if task == TEXTCAT_TASK else 5,
        )

        with gr.Row():
            sample_dataset = gr.Dataframe(
                value=default_datasets[0],
                label=(
                    "Sample dataset. Text truncated to 256 tokens."
                    if task == TEXTCAT_TASK
                    else "Sample dataset. Prompts and completions truncated to 256 tokens."
                ),
                interactive=False,
                wrap=True,
            )

        with gr.Row():
            gr.Column(scale=1)
            btn_generate_sample_dataset = gr.Button(
                value="Generate sample dataset", variant="primary"
            )
            gr.Column(scale=1)

    return (
        dataset_description,
        examples,
        btn_generate_system_prompt,
        system_prompt,
        sample_dataset,
        btn_generate_sample_dataset,
    )


def get_argilla_tab() -> Tuple[Any]:
    with gr.Tab(label="Argilla"):
        if get_argilla_client() is not None:
            with gr.Row(variant="panel"):
                dataset_name = gr.Textbox(
                    label="Dataset name",
                    placeholder="dataset_name",
                    value="my-distiset",
                )
                add_to_existing_dataset = gr.Checkbox(
                    label="Allow adding records to existing dataset",
                    info="When selected, you do need to ensure the dataset options are the same as in the existing dataset.",
                    value=False,
                    interactive=True,
                    scale=1,
                )

            with gr.Row(variant="panel"):
                btn_generate_full_dataset_argilla = gr.Button(
                    value="Generate", variant="primary", scale=2
                )
                btn_generate_and_push_to_argilla = gr.Button(
                    value="Generate and Push to Argilla",
                    variant="primary",
                    scale=2,
                )
                btn_push_to_argilla = gr.Button(
                    value="Push to Argilla", variant="primary", scale=2
                )
        else:
            gr.Markdown(
                "Please add `ARGILLA_API_URL` and `ARGILLA_API_KEY` to use Argilla or export the dataset to the Hugging Face Hub."
            )
    return (
        dataset_name,
        add_to_existing_dataset,
        btn_generate_full_dataset_argilla,
        btn_generate_and_push_to_argilla,
        btn_push_to_argilla,
    )


def get_hf_tab() -> Tuple[Any]:
    with gr.Tab("Hugging Face Hub"):
        with gr.Row(variant="panel"):
            org_name = get_org_dropdown()
            repo_name = gr.Textbox(
                label="Repo name",
                placeholder="dataset_name",
                value="my-distiset",
            )
            private = gr.Checkbox(
                label="Private dataset",
                value=True,
                interactive=True,
                scale=1,
            )
        with gr.Row(variant="panel"):
            btn_generate_full_dataset = gr.Button(
                value="Generate", variant="primary", scale=2
            )
            btn_generate_and_push_to_hub = gr.Button(
                value="Generate and Push to Hub", variant="primary", scale=2
            )
            btn_push_to_hub = gr.Button(value="Push to Hub", variant="primary", scale=2)
    return (
        org_name,
        repo_name,
        private,
        btn_generate_full_dataset,
        btn_generate_and_push_to_hub,
        btn_push_to_hub,
    )


def push_pipeline_code_to_hub(
    pipeline_code: str,
    org_name: str,
    repo_name: str,
    oauth_token: Union[OAuthToken, None] = None,
    progress=gr.Progress(),
):
    repo_id = validate_push_to_hub(org_name, repo_name)
    progress(0.1, desc="Uploading pipeline code")
    with io.BytesIO(pipeline_code.encode("utf-8")) as f:
        upload_file(
            path_or_fileobj=f,
            path_in_repo="pipeline.py",
            repo_id=repo_id,
            repo_type="dataset",
            token=oauth_token.token,
            commit_message="Include pipeline script",
            create_pr=False,
        )
    progress(1.0, desc="Pipeline code uploaded")


def push_dataset_to_hub(
    dataframe: pd.DataFrame,
    private: bool = True,
    org_name: str = None,
    repo_name: str = None,
    oauth_token: Union[OAuthToken, None] = None,
    progress=gr.Progress(),
    labels: List[str] = None,
    num_labels: int = None,
    task: str = TEXTCAT_TASK,
) -> pd.DataFrame:
    progress(0.1, desc="Setting up dataset")
    repo_id = validate_push_to_hub(org_name, repo_name)

    if task == TEXTCAT_TASK:
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
        distiset = Distiset(
            {"default": Dataset.from_pandas(dataframe, features=features)}
        )
    else:
        distiset = Distiset({"default": Dataset.from_pandas(dataframe)})
    progress(0.2, desc="Pushing dataset to hub")
    distiset.push_to_hub(
        repo_id=repo_id,
        private=private,
        include_script=False,
        token=oauth_token.token,
        create_pr=False,
    )
    progress(1.0, desc="Dataset pushed to hub")
    return dataframe


def validate_push_to_hub(org_name, repo_name):
    repo_id = (
        f"{org_name}/{repo_name}"
        if repo_name is not None and org_name is not None
        else None
    )
    if repo_id is not None:
        if not all([repo_id, org_name, repo_name]):
            raise gr.Error(
                "Please provide a `repo_name` and `org_name` to push the dataset to."
            )
    return repo_id


def get_final_dataset_row(default_datasets) -> gr.Dataframe:
    with gr.Row():
        final_dataset = gr.Dataframe(
            value=default_datasets[0],
            label="Generated dataset",
            interactive=False,
            wrap=True,
            min_width=300,
        )
    return final_dataset


def get_success_message_row() -> gr.Markdown:
    with gr.Row():
        success_message = gr.Markdown(visible=False)
    return success_message


def show_success_message(org_name, repo_name) -> gr.Markdown:
    client = get_argilla_client()
    if client is None:
        return gr.Markdown(
            value="""
            <div style="padding: 1em; background-color: #e6f3e6; border-radius: 5px; margin-top: 1em;">
                <h3 style="color: #2e7d32; margin: 0;">Dataset Published Successfully!</h3>
                <p style="margin-top: 0.5em;">
                The generated dataset is in the right format for fine-tuning with TRL, AutoTrain, or other frameworks. Your dataset is now available at:
                <a href="https://huggingface.co/datasets/{org_name}/{repo_name}" target="_blank" style="color: #1565c0; text-decoration: none;">
                    https://huggingface.co/datasets/{org_name}/{repo_name}
                    </a>
                </p>
                <p style="margin-top: 1em; font-size: 0.9em; color: #333;">
                    By configuring an `ARGILLA_API_URL` and `ARGILLA_API_KEY` you can curate the dataset in Argilla.
                    Unfamiliar with Argilla? Here are some docs to help you get started:
                    <br>• <a href="https://docs.argilla.io/latest/getting_started/quickstart/" target="_blank">How to get started with Argilla</a>
                    <br>• <a href="https://docs.argilla.io/latest/how_to_guides/annotate/" target="_blank">How to curate data in Argilla</a>
                    <br>• <a href="https://docs.argilla.io/latest/how_to_guides/import_export/" target="_blank">How to export data once you have reviewed the dataset</a>
                </p>
            </div>
            """
        )
    argilla_api_url = client.api_url
    return gr.Markdown(
        value=f"""
        <div style="padding: 1em; background-color: #e6f3e6; border-radius: 5px; margin-top: 1em;">
            <h3 style="color: #2e7d32; margin: 0;">Dataset Published Successfully!</h3>
            <p style="margin-top: 0.5em;">
                <strong>
                    <a href="{argilla_api_url}" target="_blank" style="color: #1565c0; text-decoration: none;">
                        Open your dataset in the Argilla space
                    </a>
                </strong>
            </p>
            <p style="margin-top: 0.5em;">
                The generated dataset is in the right format for fine-tuning with TRL, AutoTrain, or other frameworks. Your dataset is now available at:
                <a href="https://huggingface.co/datasets/{org_name}/{repo_name}" target="_blank" style="color: #1565c0; text-decoration: none;">
                    https://huggingface.co/datasets/{org_name}/{repo_name}
                </a>
            </p>
        </div>
        <p style="margin-top: 1em; font-size: 0.9em; color: #333;">
            Unfamiliar with Argilla? Here are some docs to help you get started:
            <br>• <a href="https://docs.argilla.io/latest/how_to_guides/annotate/" target="_blank">How to curate data in Argilla</a>
            <br>• <a href="https://docs.argilla.io/latest/how_to_guides/import_export/" target="_blank">How to export data once you have reviewed the dataset</a>
        </p>
        """,
        visible=True,
    )


def hide_success_message() -> gr.Markdown:
    return gr.Markdown(value="")
