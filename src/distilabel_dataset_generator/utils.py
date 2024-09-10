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
            value="Sign in with Hugging Face to generate a dataset!",
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
        return organisations
    else:
        return []


def get_org_dropdown(token: OAuthToken = None):
    orgs = list_orgs(token)
    return gr.Dropdown(
        label="Organization", choices=orgs, value=orgs[0] if orgs else None
    )


def swap_visibilty(profile: Union[gr.OAuthProfile, None]):
    if get_space():
        if profile is None:
            return gr.Column(visible=False)
        else:
            return gr.Column(visible=True)
    else:
        return gr.Column(visible=True)


def get_css():
    css = """
h1{font-size: 2em}
h3{margin-top: 0}
#component-1{text-align:center}
.main_ui_logged_out{opacity: 0.3; pointer-events: none}
.tabitem{border: 0px}
.group_padding{padding: .55em}
#space_model .wrap > label:last-child{opacity: 0.3; pointer-events:none}
"""
    return css
