import io
import uuid
from typing import List, Union

import argilla as rg
import gradio as gr
import pandas as pd
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from distilabel.distiset import Distiset
from gradio import OAuthToken
from huggingface_hub import HfApi, upload_file

from synthetic_dataset_generator.constants import TEXTCAT_TASK
from synthetic_dataset_generator.utils import (
    get_argilla_client,
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


def show_success_message(org_name, repo_name) -> gr.Markdown:
    client = get_argilla_client()
    if client is None:
        return gr.Markdown(
            value="""
                <div style="padding: 1em; background-color: var(--block-background-fill); border-color: var(--border-color-primary); border-width: 1px; border-radius: 5px;">
                    <h3 style="color: #2e7d32; margin: 0;">Dataset Published Successfully!</h3>
                    <p style="margin-top: 0.5em;">
                        The generated dataset is in the right format for fine-tuning with TRL, AutoTrain, or other frameworks.
                        <a href="https://huggingface.co/datasets/{org_name}/{repo_name}" target="_blank" class="lg primary svelte-cmf5ev" style="color: white !important; margin-top: 0.5em; text-decoration: none;">
                            Open in Hub
                        </a>
                    </p>
                    <p style="margin-top: 1em; color: #333;">
                        By configuring an `ARGILLA_API_URL` and `ARGILLA_API_KEY` you can curate the dataset in Argilla.
                        Unfamiliar with Argilla? Here are some docs to help you get started:
                        <br>• <a href="https://docs.argilla.io/latest/getting_started/quickstart/" target="_blank">How to get started with Argilla</a>
                        <br>• <a href="https://docs.argilla.io/latest/how_to_guides/annotate/" target="_blank">How to curate data in Argilla</a>
                        <br>• <a href="https://docs.argilla.io/latest/how_to_guides/import_export/" target="_blank">How to export data once you have reviewed the dataset</a>
                    </p>
                </div>
                """,
            visible=True,
        )
    argilla_api_url = client.api_url
    return gr.Markdown(
        value=f"""
                <div style="padding: 1em; background-color: var(--block-background-fill); border-color: var(--border-color-primary); border-width: 1px; border-radius: 5px;">
                    <h3 style="color: #2e7d32; margin: 0;">Dataset Published Successfully!</h3>
                    <p style="margin-top: 0.5em;">
                        The generated dataset is in the right format for fine-tuning with TRL, AutoTrain, or other frameworks.
                        <div style="display: flex; gap: 10px;">
                            <a href="{argilla_api_url}" target="_blank" class="lg primary svelte-cmf5ev" style="color: white !important; margin-top: 0.5em; text-decoration: none;">
                                Open in Argilla
                            </a>
                            <a href="https://huggingface.co/datasets/{org_name}/{repo_name}" target="_blank" class="lg secondary svelte-cmf5ev" style="color: black !important; margin-top: 0.5em; text-decoration: none;">
                                Open in Hub
                            </a>
                        </div>
                    </p>
                    <p style="margin-top: 1em; color: #333;">
                        Unfamiliar with Argilla? Here are some docs to help you get started:
                        <br>• <a href="https://docs.argilla.io/latest/how_to_guides/annotate/" target="_blank">How to curate data in Argilla</a>
                        <br>• <a href="https://docs.argilla.io/latest/how_to_guides/import_export/" target="_blank">How to export data once you have reviewed the dataset</a>
                    </p>
                </div>
            """,
        visible=True,
    )


def hide_success_message() -> gr.Markdown:
    return gr.Markdown(value="")
