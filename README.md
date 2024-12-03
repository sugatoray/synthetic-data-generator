---
title: Synthetic Data Generator
short_description: Build datasets using natural language
emoji: 🧬
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: true
license: apache-2.0
hf_oauth: true
#header: mini
hf_oauth_scopes:
- read-repos
- write-repos
- manage-repos
- inference-api
---
<br>

<h2 align="center">
  <a href=""><img src="https://raw.githubusercontent.com/argilla-io/synthetic-data-generator/main/assets/logo.svg" alt="Synthetic Data Generator Logo" width="80%"></a>
</h2>
<h3 align="center">Build datasets using natural language</h3>

![Synthetic Data Generator](https://huggingface.co/spaces/argilla/synthetic-data-generator/resolve/main/assets/ui-full.png)

## Introduction

Synthetic Data Generator is a tool that allows you to create high-quality datasets for training and fine-tuning language models. It leverages the power of distilabel and LLMs to generate synthetic data tailored to your specific needs.

Supported Tasks:

- Text Classification
- Supervised Fine-Tuning
- Judging and rationale evaluation

This tool simplifies the process of creating custom datasets, enabling you to:

- Describe the characteristics of your desired application
- Iterate on sample datasets
- Produce full-scale datasets
- Push your datasets to the [Hugging Face Hub](https://huggingface.co/datasets?other=datacraft) and/or [Argilla](https://docs.argilla.io/)

By using the Synthetic Data Generator, you can rapidly prototype and create datasets for, accelerating your AI development process.

<p align="center">
<a href="https://twitter.com/argilla_io">
<img src="https://img.shields.io/badge/twitter-black?logo=x"/>
</a>
<a href="https://www.linkedin.com/company/argilla-io">
<img src="https://img.shields.io/badge/linkedin-blue?logo=linkedin"/>
</a>
<a href="http://hf.co/join/discord">
<img src="https://img.shields.io/badge/Discord-7289DA?&logo=discord&logoColor=white"/>
</a>
</p>

## Installation

You can simply install the package with:

```bash
pip install synthetic-dataset-generator
```

### Quickstart

```python
from synthetic_dataset_generator.app import demo

demo.launch()
```

### Environment Variables

- `HF_TOKEN`: Your [Hugging Face token](https://huggingface.co/settings/tokens/new?ownUserPermissions=repo.content.read&ownUserPermissions=repo.write&globalPermissions=inference.serverless.write&tokenType=fineGrained) to push your datasets to the Hugging Face Hub and generate free completions from Hugging Face Inference Endpoints.

Optionally, you can set the following environment variables to customize the generation process.

- `BASE_URL`: The base URL for any OpenAI compatible API, e.g. `https://api-inference.huggingface.co/v1/`, `https://api.openai.com/v1/`.
- `MODEL`: The model to use for generating the dataset, e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`, `gpt-4o`.
- `API_KEY`: The API key to use for the corresponding API, e.g. `hf_...`, `sk-...`.

Optionally, you can also push your datasets to Argilla for further curation by setting the following environment variables:

- `ARGILLA_API_KEY`: Your Argilla API key to push your datasets to Argilla.
- `ARGILLA_API_URL`: Your Argilla API URL to push your datasets to Argilla.

### Argilla integration

Argilla is a open source tool for data curation. It allows you to annotate and review datasets, and push curated datasets to the Hugging Face Hub. You can easily get started with Argilla by following the [quickstart guide](https://docs.argilla.io/latest/getting_started/quickstart/).

![Argilla integration](https://huggingface.co/spaces/argilla/synthetic-data-generator/resolve/main/assets/argilla.png)

## Custom synthetic data generation?

Each pipeline is based on distilabel, so you can easily change the LLM or the pipeline steps.

Check out the [distilabel library](https://github.com/argilla-io/distilabel) for more information.

## Development

Install the dependencies:

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the dependencies
pip install -e . # pdm install
```

Run the app:

```bash
python app.py
```
