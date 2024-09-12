import gradio as gr
from gradio.oauth import (
    OAUTH_CLIENT_ID,
    OAUTH_CLIENT_SECRET,
    OAUTH_SCOPES,
    OPENID_PROVIDER_URL,
    get_space,
)
from huggingface_hub import whoami

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
    return gr.LoginButton(
        value="Sign in with Hugging Face!",
        size="lg",
    ).activate()


def get_duplicate_button():
    if get_space() is not None:
        return gr.DuplicateButton(size="lg")


def list_orgs(oauth_token: OAuthToken = None):
    if oauth_token is None:
        return []
    data = whoami(oauth_token.token)
    if data["auth"]["type"] == "oauth":
        organisations = [data["name"]] + [org["name"] for org in data["orgs"]]
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
    orgs = list_orgs(oauth_token)
    return gr.Dropdown(
        label="Organization",
        choices=orgs,
        value=orgs[0] if orgs else None,
        allow_custom_value=True,
    )


def get_token(oauth_token: OAuthToken = None):
    if oauth_token:
        return oauth_token.token
    else:
        return ""


def swap_visibilty(oauth_token: OAuthToken = None):
    if oauth_token is None:
        return gr.update(elem_classes=["main_ui_logged_out"])
    else:
        return gr.update(elem_classes=["main_ui_logged_in"])
