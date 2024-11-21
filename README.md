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

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

<div class="header-container">
    <div class="title-container">
        <h1 style="margin: 0; font-size: 2em;">🧬 Synthetic Data Generator</h1>
        <p style="margin: 10px 0 0 0; color: #666; font-size: 1.1em;">Build datasets using natural language</p>
    </div>
</div>
<br>

This repository contains the code for the [free Synthetic Data Generator app](https://huggingface.co/spaces/argilla/synthetic-data-generator), which is hosted on the Hugging Face Hub.

## How it works?

![Synthetic Data Generator](https://huggingface.co/spaces/argilla/synthetic-data-generator/resolve/main/assets/flow.png)

Distilabel Synthetic Data Generator is a tool that allows you to easily create high-quality datasets for training and fine-tuning language models. It leverages the power of distilabel and advanced language models to generate synthetic data tailored to your specific needs.

This tool simplifies the process of creating custom datasets, enabling you to:

- Define the characteristics of your desired application
- Generate system prompts and tasks automatically
- Create sample datasets for quick iteration
- Produce full-scale datasets with customizable parameters
- Push your generated datasets directly to the Hugging Face Hub

By using Distilabel Synthetic Data Generator, you can rapidly prototype and create datasets for, accelerating your AI development process.

## Do you want to run this locally?

You can simply clone the repository and run it locally with:

```bash
pip install -r requirements.txt
python app.py
```

Note that you do need to have an `HF_TOKEN` that can make calls to the free serverless Hugging Face Inference Endpoints. You can get one [here](https://huggingface.co/settings/tokens/new?ownUserPermissions=repo.content.read&ownUserPermissions=repo.write&globalPermissions=inference.serverless.write&tokenType=fineGrained).

## Do you need more control?

Each pipeline is based on a distilabel component, so you can easily run it locally or with other LLMs.

Check out the [distilabel library](https://github.com/argilla-io/distilabel) for more information.
