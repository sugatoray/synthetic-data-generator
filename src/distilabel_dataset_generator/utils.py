from typing import Union

import gradio as gr
from gradio.oauth import (
    OAUTH_CLIENT_ID,
    OAUTH_CLIENT_SECRET,
    OAUTH_SCOPES,
    OPENID_PROVIDER_URL,
    get_space,
)
from huggingface_hub import whoami

if (
    all(
        [
            OAUTH_CLIENT_ID,
            OAUTH_CLIENT_SECRET,
            OAUTH_SCOPES,
            OPENID_PROVIDER_URL,
        ]
    )
    or get_space() is None
):
    from gradio.oauth import OAuthToken
else:
    OAuthToken = str


def get_login_button():
    if (
        all(
            [
                OAUTH_CLIENT_ID,
                OAUTH_CLIENT_SECRET,
                OAUTH_SCOPES,
                OPENID_PROVIDER_URL,
            ]
        )
        or get_space() is None
    ):
        return gr.LoginButton(
            value="Sign in with Hugging Face to generate a full dataset and push it to the Hub!",
            size="lg",
        )


def get_duplicate_button():
    if get_space() is not None:
        return gr.DuplicateButton(size="lg")


def list_orgs(token: OAuthToken = None):
    if token is not None:
        data = whoami(token)
        organisations = [
            entry["entity"]["name"]
            for entry in data["auth"]["accessToken"]["fineGrained"]["scoped"]
            if "repo.write" in entry["permissions"]
        ]
        organisations.append(data["name"])
        return list(set(organisations))
    else:
        return []


def get_org_dropdown(token: OAuthToken = None):
    orgs = list_orgs(token)
    return gr.Dropdown(
        label="Organization", choices=orgs, value=orgs[0] if orgs else None
    )


def swap_visibilty(profile: Union[gr.OAuthProfile, None]):
    if profile is None:
        return gr.Column(visible=False)
    else:
        return gr.Column(visible=True)
