import gradio as gr
import requests

with gr.Blocks() as app:
    with gr.Row():
        url = "https://raw.githubusercontent.com/argilla-io/synthetic-data-generator/refs/heads/main/README.md"
        response = requests.get(url)
        readme_content: str = response.text
        readme_content = readme_content.split("## Introduction")[1]
        readme_content = "## Introduction" + readme_content
        gr.Markdown(readme_content)
