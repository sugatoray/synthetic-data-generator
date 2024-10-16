import ast
import io
from typing import Dict, List, Union

import argilla as rg
import gradio as gr
import pandas as pd
from datasets import Dataset
from distilabel.distiset import Distiset
from distilabel.steps.tasks.text_generation import TextGeneration
from gradio.oauth import OAuthToken
from huggingface_hub import upload_file
from huggingface_hub.hf_api import HfApi

from src.distilabel_dataset_generator.pipelines.embeddings import (
    get_embeddings,
    get_sentence_embedding_dimensions,
)
from src.distilabel_dataset_generator.pipelines.sft import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_DESCRIPTIONS,
    DEFAULT_DATASETS,
    DEFAULT_SYSTEM_PROMPTS,
    PROMPT_CREATION_PROMPT,
    generate_pipeline_code,
    get_magpie_generator,
    get_prompt_generator,
    get_response_generator,
)
from src.distilabel_dataset_generator.utils import (
    get_argilla_client,
    get_login_button,
    get_org_dropdown,
    swap_visibilty,
)


def convert_to_list_of_dicts(messages: str) -> List[Dict[str, str]]:
    return ast.literal_eval(
        messages.replace("'user'}", "'user'},")
        .replace("'system'}", "'system'},")
        .replace("'assistant'}", "'assistant'},")
    )


def generate_system_prompt(dataset_description, progress=gr.Progress()):
    progress(0.0, desc="Generating system prompt")
    if dataset_description in DEFAULT_DATASET_DESCRIPTIONS:
        index = DEFAULT_DATASET_DESCRIPTIONS.index(dataset_description)
        if index < len(DEFAULT_SYSTEM_PROMPTS):
            return DEFAULT_SYSTEM_PROMPTS[index]

    progress(0.3, desc="Initializing text generation")
    generate_description: TextGeneration = get_prompt_generator()
    progress(0.7, desc="Generating system prompt")
    result = next(
        generate_description.process(
            [
                {
                    "system_prompt": PROMPT_CREATION_PROMPT,
                    "instruction": dataset_description,
                }
            ]
        )
    )[0]["generation"]
    progress(1.0, desc="System prompt generated")
    return result


def generate_sample_dataset(system_prompt, progress=gr.Progress()):
    if system_prompt in DEFAULT_SYSTEM_PROMPTS:
        index = DEFAULT_SYSTEM_PROMPTS.index(system_prompt)
        if index < len(DEFAULT_DATASETS):
            return DEFAULT_DATASETS[index]
    result = generate_dataset(
        system_prompt, num_turns=1, num_rows=1, progress=progress, is_sample=True
    )
    return result


def _check_push_to_hub(org_name, repo_name):
    repo_id = (
        f"{org_name}/{repo_name}"
        if repo_name is not None and org_name is not None
        else None
    )
    if repo_id is not None:
        if not all([repo_id, org_name, repo_name]):
            raise gr.Error(
                "Please provide a `repo_name` and `org_name` to push the dataset to."
            )
    return repo_id


def generate_dataset(
    system_prompt: str,
    num_turns: int = 1,
    num_rows: int = 5,
    is_sample: bool = False,
    progress=gr.Progress(),
) -> pd.DataFrame:
    progress(0.0, desc="(1/2) Generating instructions")
    magpie_generator = get_magpie_generator(
        num_turns, num_rows, system_prompt, is_sample
    )
    response_generator = get_response_generator(num_turns, system_prompt, is_sample)
    total_steps: int = num_rows * 2
    batch_size = DEFAULT_BATCH_SIZE

    # create instructions
    n_processed = 0
    magpie_results = []
    while n_processed < num_rows:
        progress(
            0.5 * n_processed / num_rows,
            total=total_steps,
            desc="(1/2) Generating instructions",
        )
        remaining_rows = num_rows - n_processed
        batch_size = min(batch_size, remaining_rows)
        inputs = [{"system_prompt": system_prompt} for _ in range(batch_size)]
        batch = list(magpie_generator.process(inputs=inputs))
        magpie_results.extend(batch[0])
        n_processed += batch_size
    progress(0.5, desc="(1/2) Generating instructions")

    # generate responses
    n_processed = 0
    response_results = []
    if num_turns == 1:
        while n_processed < num_rows:
            progress(
                0.5 + 0.5 * n_processed / num_rows,
                total=total_steps,
                desc="(2/2) Generating responses",
            )
            batch = magpie_results[n_processed : n_processed + batch_size]
            responses = list(response_generator.process(inputs=batch))
            response_results.extend(responses[0])
            n_processed += batch_size
        for result in response_results:
            result["prompt"] = result["instruction"]
            result["completion"] = result["generation"]
            result["system_prompt"] = system_prompt
    else:
        for result in magpie_results:
            result["conversation"].insert(
                0, {"role": "system", "content": system_prompt}
            )
            result["messages"] = result["conversation"]
        while n_processed < num_rows:
            progress(
                0.5 + 0.5 * n_processed / num_rows,
                total=total_steps,
                desc="(2/2) Generating responses",
            )
            batch = magpie_results[n_processed : n_processed + batch_size]
            responses = list(response_generator.process(inputs=batch))
            response_results.extend(responses[0])
            n_processed += batch_size
        for result in response_results:
            result["messages"].append(
                {"role": "assistant", "content": result["generation"]}
            )
    progress(
        1,
        total=total_steps,
        desc="(2/2) Generating responses",
    )

    # create distiset
    distiset_results = []
    for result in response_results:
        record = {}
        for relevant_keys in [
            "messages",
            "prompt",
            "completion",
            "model_name",
            "system_prompt",
        ]:
            if relevant_keys in result:
                record[relevant_keys] = result[relevant_keys]
        distiset_results.append(record)

    distiset = Distiset(
        {
            "default": Dataset.from_list(distiset_results),
        }
    )

    # If not pushing to hub generate the dataset directly
    distiset = distiset["default"]
    if num_turns == 1:
        outputs = distiset.to_pandas()[["system_prompt", "prompt", "completion"]]
    else:
        outputs = distiset.to_pandas()[["messages"]]
    dataframe = pd.DataFrame(outputs)
    progress(1.0, desc="Dataset generation completed")
    return dataframe


def push_to_hub(
    dataframe: pd.DataFrame,
    private: bool = True,
    org_name: str = None,
    repo_name: str = None,
    oauth_token: Union[OAuthToken, None] = None,
    progress=gr.Progress(),
) -> pd.DataFrame:
    original_dataframe = dataframe.copy(deep=True)
    if "messages" in dataframe.columns:
        dataframe["messages"] = dataframe["messages"].apply(
            lambda x: convert_to_list_of_dicts(x) if isinstance(x, str) else x
        )
    progress(0.1, desc="Setting up dataset")
    repo_id = _check_push_to_hub(org_name, repo_name)
    distiset = Distiset(
        {
            "default": Dataset.from_pandas(dataframe),
        }
    )
    progress(0.2, desc="Pushing dataset to hub")
    distiset.push_to_hub(
        repo_id=repo_id,
        private=private,
        include_script=False,
        token=oauth_token.token,
        create_pr=False,
    )
    progress(1.0, desc="Dataset pushed to hub")
    return original_dataframe


def push_to_argilla(
    dataframe: pd.DataFrame,
    dataset_name: str,
    oauth_token: Union[OAuthToken, None] = None,
    progress=gr.Progress(),
) -> pd.DataFrame:
    original_dataframe = dataframe.copy(deep=True)
    if "messages" in dataframe.columns:
        dataframe["messages"] = dataframe["messages"].apply(
            lambda x: convert_to_list_of_dicts(x) if isinstance(x, str) else x
        )
    try:
        progress(0.1, desc="Setting up user and workspace")
        client = get_argilla_client()
        hf_user = HfApi().whoami(token=oauth_token.token)["name"]

        # Create user if it doesn't exist
        rg_user = client.users(username=hf_user)
        if rg_user is None:
            rg_user = client.users.add(rg.User(username=hf_user, role="admin"))

        # Create workspace if it doesn't exist
        workspace = client.workspaces(name=rg_user.username)
        if workspace is None:
            workspace = client.workspaces.add(rg.Workspace(name=rg_user.username))
            workspace.add_user(rg_user)

        if "messages" in dataframe.columns:
            settings = rg.Settings(
                fields=[
                    rg.ChatField(
                        name="messages",
                        description="The messages in the conversation",
                        title="Messages",
                    ),
                ],
                questions=[
                    rg.RatingQuestion(
                        name="rating",
                        title="Rating",
                        description="The rating of the conversation",
                        values=list(range(1, 6)),
                    ),
                ],
                metadata=[
                    rg.IntegerMetadataProperty(
                        name="user_message_length", title="User Message Length"
                    ),
                    rg.IntegerMetadataProperty(
                        name="assistant_message_length",
                        title="Assistant Message Length",
                    ),
                ],
                vectors=[
                    rg.VectorField(
                        name="messages_embeddings",
                        dimensions=get_sentence_embedding_dimensions(),
                    )
                ],
                guidelines="Please review the conversation and provide a score for the assistant's response.",
            )

            dataframe["user_message_length"] = dataframe["messages"].apply(
                lambda x: sum([len(y["content"]) for y in x if y["role"] == "user"])
            )
            dataframe["assistant_message_length"] = dataframe["messages"].apply(
                lambda x: sum(
                    [len(y["content"]) for y in x if y["role"] == "assistant"]
                )
            )
            dataframe["messages_embeddings"] = get_embeddings(
                dataframe["messages"].apply(
                    lambda x: " ".join([y["content"] for y in x])
                )
            )
        else:
            settings = rg.Settings(
                fields=[
                    rg.TextField(
                        name="system_prompt",
                        title="System Prompt",
                        description="The system prompt used for the conversation",
                        required=False,
                    ),
                    rg.TextField(
                        name="prompt",
                        title="Prompt",
                        description="The prompt used for the conversation",
                    ),
                    rg.TextField(
                        name="completion",
                        title="Completion",
                        description="The completion from the assistant",
                    ),
                ],
                questions=[
                    rg.RatingQuestion(
                        name="rating",
                        title="Rating",
                        description="The rating of the conversation",
                        values=list(range(1, 6)),
                    ),
                ],
                metadata=[
                    rg.IntegerMetadataProperty(
                        name="prompt_length", title="Prompt Length"
                    ),
                    rg.IntegerMetadataProperty(
                        name="completion_length", title="Completion Length"
                    ),
                ],
                vectors=[
                    rg.VectorField(
                        name="prompt_embeddings",
                        dimensions=get_sentence_embedding_dimensions(),
                    )
                ],
                guidelines="Please review the conversation and correct the prompt and completion where needed.",
            )
            dataframe["prompt_length"] = dataframe["prompt"].apply(len)
            dataframe["completion_length"] = dataframe["completion"].apply(len)
            dataframe["prompt_embeddings"] = get_embeddings(dataframe["prompt"])

        progress(0.5, desc="Creating dataset")
        rg_dataset = client.datasets(name=dataset_name, workspace=rg_user.username)
        if rg_dataset is None:
            rg_dataset = rg.Dataset(
                name=dataset_name,
                workspace=rg_user.username,
                settings=settings,
                client=client,
            )
            rg_dataset = rg_dataset.create()
        progress(0.7, desc="Pushing dataset to Argilla")
        hf_dataset = Dataset.from_pandas(dataframe)
        rg_dataset.records.log(records=hf_dataset)
        progress(1.0, desc="Dataset pushed to Argilla")
    except Exception as e:
        raise gr.Error(f"Error pushing dataset to Argilla: {e}")
    return original_dataframe


def validate_argilla_dataset_name(
    dataset_name: str,
    final_dataset: pd.DataFrame,
    add_to_existing_dataset: bool,
    oauth_token: Union[OAuthToken, None] = None,
    progress=gr.Progress(),
) -> str:
    progress(0, desc="Validating dataset configuration")
    hf_user = HfApi().whoami(token=oauth_token.token)["name"]
    client = get_argilla_client()
    if dataset_name is None or dataset_name == "":
        raise gr.Error("Dataset name is required")
    dataset = client.datasets(name=dataset_name, workspace=hf_user)
    if dataset and not add_to_existing_dataset:
        raise gr.Error(f"Dataset {dataset_name} already exists")
    return final_dataset


def upload_pipeline_code(
    pipeline_code,
    org_name,
    repo_name,
    oauth_token: Union[OAuthToken, None] = None,
    progress=gr.Progress(),
):
    repo_id = _check_push_to_hub(org_name, repo_name)
    progress(0.1, desc="Uploading pipeline code")
    with io.BytesIO(pipeline_code.encode("utf-8")) as f:
        upload_file(
            path_or_fileobj=f,
            path_in_repo="pipeline.py",
            repo_id=repo_id,
            repo_type="dataset",
            token=oauth_token.token,
            commit_message="Include pipeline script",
            create_pr=False,
        )
    progress(1.0, desc="Pipeline code uploaded")


css = """
.main_ui_logged_out{opacity: 0.3; pointer-events: none}
"""

with gr.Blocks(
    title="🧬 Synthetic Data Generator",
    head="🧬  Synthetic Data Generator",
    css=css,
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
        dataset_description = gr.TextArea(
            label="Give a precise description of the assistant or tool. Don't describe the dataset",
            value=DEFAULT_DATASET_DESCRIPTIONS[0],
            lines=2,
        )
        examples = gr.Examples(
            elem_id="system_prompt_examples",
            examples=[[example] for example in DEFAULT_DATASET_DESCRIPTIONS],
            inputs=[dataset_description],
        )
        with gr.Row():
            gr.Column(scale=1)
            btn_generate_system_prompt = gr.Button(
                value="Generate system prompt and sample dataset"
            )
            gr.Column(scale=1)

        system_prompt = gr.TextArea(
            label="System prompt for dataset generation. You can tune it and regenerate the sample",
            value=DEFAULT_SYSTEM_PROMPTS[0],
            lines=5,
        )

        with gr.Row():
            sample_dataset = gr.Dataframe(
                value=DEFAULT_DATASETS[0],
                label="Sample dataset. Prompts and completions truncated to 256 tokens.",
                interactive=False,
                wrap=True,
            )

        with gr.Row():
            gr.Column(scale=1)
            btn_generate_sample_dataset = gr.Button(
                value="Generate sample dataset",
            )
            gr.Column(scale=1)

        result = btn_generate_system_prompt.click(
            fn=generate_system_prompt,
            inputs=[dataset_description],
            outputs=[system_prompt],
            show_progress=True,
        ).then(
            fn=generate_sample_dataset,
            inputs=[system_prompt],
            outputs=[sample_dataset],
            show_progress=True,
        )

        btn_generate_sample_dataset.click(
            fn=generate_sample_dataset,
            inputs=[system_prompt],
            outputs=[sample_dataset],
            show_progress=True,
        )

        # Add a header for the full dataset generation section
        gr.Markdown("## Generate full dataset")
        gr.Markdown(
            "Once you're satisfied with the sample, generate a larger dataset and push it to Argilla or the Hugging Face Hub."
        )

        with gr.Column() as push_to_hub_ui:
            with gr.Row(variant="panel"):
                num_turns = gr.Number(
                    value=1,
                    label="Number of turns in the conversation",
                    minimum=1,
                    maximum=4,
                    step=1,
                    info="Choose between 1 (single turn with 'instruction-response' columns) and 2-4 (multi-turn conversation with a 'messages' column).",
                )
                num_rows = gr.Number(
                    value=10,
                    label="Number of rows in the dataset",
                    minimum=1,
                    maximum=500,
                    info="The number of rows in the dataset. Note that you are able to generate more rows at once but that this will take time.",
                )

            with gr.Tab(label="Argilla"):
                if get_argilla_client():
                    with gr.Row(variant="panel"):
                        dataset_name = gr.Textbox(
                            label="Dataset name",
                            placeholder="dataset_name",
                            value="my-distiset",
                        )
                        add_to_existing_dataset = gr.Checkbox(
                            label="Allow adding records to existing dataset",
                            info="When selected, you do need to ensure the number of turns in the conversation is the same as the number of turns in the existing dataset.",
                            value=False,
                            interactive=True,
                            scale=0.5,
                        )

                    with gr.Row(variant="panel"):
                        btn_generate_full_dataset_copy = gr.Button(
                            value="Generate", variant="primary", scale=2
                        )
                        btn_generate_and_push_to_argilla = gr.Button(
                            value="Generate and Push to Argilla",
                            variant="primary",
                            scale=2,
                        )
                        btn_push_to_argilla = gr.Button(
                            value="Push to Argilla", variant="primary", scale=2
                        )
                else:
                    gr.Markdown(
                        "Please add `ARGILLA_API_URL` and `ARGILLA_API_KEY` to use Argilla."
                    )
            with gr.Tab("Hugging Face Hub"):
                with gr.Row(variant="panel"):
                    org_name = get_org_dropdown()
                    repo_name = gr.Textbox(
                        label="Repo name",
                        placeholder="dataset_name",
                        value="my-distiset",
                    )
                    private = gr.Checkbox(
                        label="Private dataset",
                        value=True,
                        interactive=True,
                        scale=0.5,
                    )
                with gr.Row(variant="panel"):
                    btn_generate_full_dataset = gr.Button(
                        value="Generate", variant="primary", scale=2
                    )
                    btn_generate_and_push_to_hub = gr.Button(
                        value="Generate and Push to Hub", variant="primary", scale=2
                    )
                    btn_push_to_hub = gr.Button(
                        value="Push to Hub", variant="primary", scale=2
                    )

            with gr.Row():
                final_dataset = gr.Dataframe(
                    value=DEFAULT_DATASETS[0],
                    label="Generated dataset",
                    interactive=False,
                    wrap=True,
                )

            with gr.Row():
                success_message = gr.Markdown(visible=False)

    def show_success_message_argilla():
        client = get_argilla_client()
        argilla_api_url = client.api_url
        return gr.Markdown(
            value=f"""
            <div style="padding: 1em; background-color: #e6f3e6; border-radius: 5px; margin-top: 1em;">
                <h3 style="color: #2e7d32; margin: 0;">Dataset Published Successfully!</h3>
                <p style="margin-top: 0.5em;">
                    Your dataset is now available at:
                    <a href="{argilla_api_url}" target="_blank" style="color: #1565c0; text-decoration: none;">
                        {argilla_api_url}
                    </a>
                </p>
            </div>
            """,
            visible=True,
        )

    def show_success_message_hub(org_name, repo_name):
        return gr.Markdown(
            value=f"""
            <div style="padding: 1em; background-color: #e6f3e6; border-radius: 5px; margin-top: 1em;">
                <h3 style="color: #2e7d32; margin: 0;">Dataset Published Successfully!</h3>
                <p style="margin-top: 0.5em;">
                    The generated dataset is in the right format for fine-tuning with TRL, AutoTrain or other frameworks.
                    Your dataset is now available at:
                    <a href="https://huggingface.co/datasets/{org_name}/{repo_name}" target="_blank" style="color: #1565c0; text-decoration: none;">
                        https://huggingface.co/datasets/{org_name}/{repo_name}
                    </a>
                </p>
            </div>
            """,
            visible=True,
        )

    def hide_success_message():
        return gr.Markdown(visible=False)

    gr.Markdown("## Or run this pipeline locally with distilabel")
    gr.Markdown(
        "You can run this pipeline locally with distilabel. For more information, please refer to the [distilabel documentation](https://distilabel.argilla.io/) or go to the FAQ tab at the top of the page for more information."
    )

    with gr.Accordion(
        "Run this pipeline using distilabel",
        open=False,
    ):
        pipeline_code = gr.Code(
            value=generate_pipeline_code(
                system_prompt.value, num_turns.value, num_rows.value
            ),
            language="python",
            label="Distilabel Pipeline Code",
        )

    sample_dataset.change(
        fn=lambda x: x,
        inputs=[sample_dataset],
        outputs=[final_dataset],
    )
    gr.on(
        triggers=[
            btn_generate_full_dataset.click,
            btn_generate_full_dataset_copy.click,
        ],
        fn=hide_success_message,
        outputs=[success_message],
    ).then(
        fn=generate_dataset,
        inputs=[system_prompt, num_turns, num_rows],
        outputs=[final_dataset],
        show_progress=True,
    )

    btn_generate_and_push_to_argilla.click(
        fn=validate_argilla_dataset_name,
        inputs=[dataset_name, final_dataset, add_to_existing_dataset],
        outputs=[final_dataset],
        show_progress=True,
    ).success(
        fn=hide_success_message,
        outputs=[success_message],
    ).success(
        fn=generate_dataset,
        inputs=[system_prompt, num_turns, num_rows],
        outputs=[final_dataset],
        show_progress=True,
    ).success(
        fn=push_to_argilla,
        inputs=[final_dataset, dataset_name],
        outputs=[final_dataset],
        show_progress=True,
    ).success(
        fn=show_success_message_argilla,
        inputs=[],
        outputs=[success_message],
    )

    btn_generate_and_push_to_hub.click(
        fn=hide_success_message,
        outputs=[success_message],
    ).then(
        fn=generate_dataset,
        inputs=[system_prompt, num_turns, num_rows],
        outputs=[final_dataset],
        show_progress=True,
    ).then(
        fn=push_to_hub,
        inputs=[final_dataset, private, org_name, repo_name],
        outputs=[final_dataset],
        show_progress=True,
    ).then(
        fn=upload_pipeline_code,
        inputs=[pipeline_code, org_name, repo_name],
        outputs=[],
        show_progress=True,
    ).success(
        fn=show_success_message_hub,
        inputs=[org_name, repo_name],
        outputs=[success_message],
    )

    btn_push_to_hub.click(
        fn=hide_success_message,
        outputs=[success_message],
    ).then(
        fn=push_to_hub,
        inputs=[final_dataset, private, org_name, repo_name],
        outputs=[final_dataset],
        show_progress=True,
    ).then(
        fn=upload_pipeline_code,
        inputs=[pipeline_code, org_name, repo_name],
        outputs=[],
        show_progress=True,
    ).success(
        fn=show_success_message_hub,
        inputs=[org_name, repo_name],
        outputs=[success_message],
    )

    btn_push_to_argilla.click(
        fn=hide_success_message,
        outputs=[success_message],
    ).success(
        fn=validate_argilla_dataset_name,
        inputs=[dataset_name, final_dataset, add_to_existing_dataset],
        outputs=[final_dataset],
        show_progress=True,
    ).success(
        fn=push_to_argilla,
        inputs=[final_dataset, dataset_name],
        outputs=[final_dataset],
        show_progress=True,
    ).success(
        fn=show_success_message_argilla,
        inputs=[],
        outputs=[success_message],
    )

    system_prompt.change(
        fn=generate_pipeline_code,
        inputs=[system_prompt, num_turns, num_rows],
        outputs=[pipeline_code],
    )
    num_turns.change(
        fn=generate_pipeline_code,
        inputs=[system_prompt, num_turns, num_rows],
        outputs=[pipeline_code],
    )
    num_rows.change(
        fn=generate_pipeline_code,
        inputs=[system_prompt, num_turns, num_rows],
        outputs=[pipeline_code],
    )
    app.load(get_org_dropdown, outputs=[org_name])
    app.load(fn=swap_visibilty, outputs=main_ui)
