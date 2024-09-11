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
            value="Sign in with Hugging Face! (This resets the session state.)",
            size="lg",
        )


def get_duplicate_button():
    if get_space() is not None:
        return gr.DuplicateButton(size="lg")


def list_orgs(oauth_token: OAuthToken = None):
    if oauth_token is None:
        return []
    data = whoami(oauth_token.token)
    organisations = [
        entry["entity"]["name"]
        for entry in data["auth"]["accessToken"]["fineGrained"]["scoped"]
        if "repo.write" in entry["permissions"]
    ]
    organisations.append(data["name"])
    return list(set(organisations))


def get_org_dropdown(token: OAuthToken = None):
    orgs = list_orgs(token)
    return gr.Dropdown(
        label="Organization",
        choices=orgs,
        value=orgs[0] if orgs else None,
        allow_custom_value=True,
    )


def get_token(token: OAuthToken = None):
    if token:
        return token.token
    else:
        return ""
