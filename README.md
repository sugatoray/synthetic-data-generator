---
title: Synthetic Data Generator
short_description: Build datasets using natural language
emoji: 🧬
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 5.5.0
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
    <div class="logo-container">
        <a href="https://github.com/argilla-io/distilabel" target="_blank" rel="noopener noreferrer">
            <img src="https://distilabel.argilla.io/latest/assets/distilabel-black.svg" alt="Distilabel Logo" style="width: 150px; height: auto;">
        </a>
    </div>
    <div class="title-container">
        <h1 style="margin: 0; font-size: 2em;">🧬 Synthetic Data Generator</h1>
        <p style="margin: 10px 0 0 0; color: #666; font-size: 1.1em;">Build datasets using natural language</p>
    </div>
</div>
<br>
This repository contains the code for the [free Synthetic Data Generator app](https://huggingface.co/spaces/argilla/synthetic-data-generator), which is hosted on the Hugging Face Hub.

## How it works?

![Synthetic Data Generator](https://huggingface.co/spaces/argilla/synthetic-data-generator/resolve/main/assets/flow.png)

Distilabel Synthetic Data Generator is an experimental tool that allows you to easily create high-quality datasets for training and fine-tuning language models. It leverages the power of distilabel and advanced language models to generate synthetic data tailored to your specific needs.

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

## Do you need more control?

Each pipeline is based on a distilabel component, so you can easily run it locally or with other LLMs.

Check out the [distilabel library](https://github.com/argilla-io/distilabel) for more information.
