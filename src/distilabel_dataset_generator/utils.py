import gradio as gr
from gradio.oauth import (
    OAUTH_CLIENT_ID,
    OAUTH_CLIENT_SECRET,
    OAUTH_SCOPES,
    OPENID_PROVIDER_URL,
    get_space,
)

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
            value="Sign in with Hugging Face - a login will reset the data!",
            size="lg",
        )
