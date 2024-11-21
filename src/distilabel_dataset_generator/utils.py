import os
from typing import List, Optional, Union

import argilla as rg
import gradio as gr
from gradio.oauth import (
    OAUTH_CLIENT_ID,
    OAUTH_CLIENT_SECRET,
    OAUTH_SCOPES,
    OPENID_PROVIDER_URL,
    get_space,
)
from huggingface_hub import whoami

_LOGGED_OUT_CSS = ".main_ui_logged_out{opacity: 0.3; pointer-events: none}"

HF_TOKENS = [os.getenv("HF_TOKEN")] + [os.getenv(f"HF_TOKEN_{i}") for i in range(1, 10)]
HF_TOKENS = [token for token in HF_TOKENS if token]

_CHECK_IF_SPACE_IS_SET = (
    all(
        [
            OAUTH_CLIENT_ID,
            OAUTH_CLIENT_SECRET,
            OAUTH_SCOPES,
            OPENID_PROVIDER_URL,
        ]
    )
    or get_space() is None
)

if _CHECK_IF_SPACE_IS_SET:
    from gradio.oauth import OAuthToken
else:
    OAuthToken = str


def get_login_button():
    return gr.LoginButton(value="Sign in!", size="sm", scale=2).activate()


def get_duplicate_button():
    if get_space() is not None:
        return gr.DuplicateButton(size="lg")


def list_orgs(oauth_token: OAuthToken = None):
    if oauth_token is None:
        return []
    data = whoami(oauth_token.token)
    if data["auth"]["type"] == "oauth":
        organisations = [data["name"]] + [org["name"] for org in data["orgs"]]
    elif data["auth"]["type"] == "access_token":
        organisations = [org["name"] for org in data["orgs"]]
    else:
        organisations = [
            entry["entity"]["name"]
            for entry in data["auth"]["accessToken"]["fineGrained"]["scoped"]
            if "repo.write" in entry["permissions"]
        ]
        organisations = [org for org in organisations if org != data["name"]]
        organisations = [data["name"]] + organisations
    return organisations


def get_org_dropdown(oauth_token: OAuthToken = None):
    if oauth_token:
        orgs = list_orgs(oauth_token)
    else:
        orgs = []
    return gr.Dropdown(
        label="Organization",
        choices=orgs,
        value=orgs[0] if orgs else None,
        allow_custom_value=True,
        interactive=True,
    )


def get_token(oauth_token: OAuthToken = None):
    if oauth_token:
        return oauth_token.token
    else:
        return ""


def swap_visibilty(oauth_token: Optional[OAuthToken] = None):
    if oauth_token:
        return gr.update(elem_classes=["main_ui_logged_in"])
    else:
        return gr.update(elem_classes=["main_ui_logged_out"])


def get_base_app():
    with gr.Blocks(
        title="🧬 Synthetic Data Generator",
        head="🧬  Synthetic Data Generator",
        css=_LOGGED_OUT_CSS,
    ) as app:
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
            pass

    return app


def get_argilla_client() -> Union[rg.Argilla, None]:
    try:
        api_url = os.getenv("ARGILLA_API_URL_SDG_REVIEWER")
        api_key = os.getenv("ARGILLA_API_KEY_SDG_REVIEWER")
        if api_url is None or api_key is None:
            api_url = os.getenv("ARGILLA_API_URL")
            api_key = os.getenv("ARGILLA_API_KEY")
        return rg.Argilla(
            api_url=api_url,
            api_key=api_key,
        )
    except Exception:
        return None


def get_preprocess_labels(labels: Optional[List[str]]) -> List[str]:
    return list(set([label.lower().strip() for label in labels])) if labels else []
