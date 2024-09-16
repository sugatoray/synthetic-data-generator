import multiprocessing
import time

import gradio as gr
import pandas as pd
from distilabel.distiset import Distiset

from src.distilabel_dataset_generator.pipelines.sft import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_DESCRIPTIONS,
    DEFAULT_SYSTEM_PROMPT,
    PROMPT_CREATION_PROMPT,
    generate_pipeline_code,
    get_pipeline,
    get_prompt_generation_step,
)
from src.distilabel_dataset_generator.utils import (
    get_login_button,
    get_org_dropdown,
    get_token,
    swap_visibilty,
)


def _run_pipeline(result_queue, num_turns, num_rows, system_prompt, is_sample):
    pipeline = get_pipeline(num_turns, num_rows, system_prompt, is_sample)
    distiset: Distiset = pipeline.run(use_cache=True)
    result_queue.put(distiset)


def generate_system_prompt(dataset_description, progress=gr.Progress()):
    progress(0.1, desc="Initializing text generation")
    generate_description = get_prompt_generation_step()
    progress(0.4, desc="Loading model")
    generate_description.load()
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
    progress(0.1, desc="Initializing sample dataset generation")
    result = generate_dataset(
        system_prompt, num_turns=1, num_rows=1, progress=progress, is_sample=True
    )
    progress(1.0, desc="Sample dataset generated")
    return result


def generate_dataset(
    system_prompt: str,
    num_turns: int = 1,
    num_rows: int = 5,
    private: bool = True,
    org_name: str = None,
    repo_name: str = None,
    oauth_token: str = None,
    progress=gr.Progress(),
    is_sample: bool = False,
):
    repo_id = (
        f"{org_name}/{repo_name}"
        if repo_name is not None and org_name is not None
        else None
    )
    if repo_id is not None:
        if not all([repo_id, org_name, repo_name]):
            raise gr.Error(
                "Please provide a repo_name and org_name to push the dataset to."
            )

    if num_turns > 4:
        num_turns = 4
        gr.Info("You can only generate a dataset with 4 or fewer turns. Setting to 4.")
    if num_rows > 5000:
        num_rows = 1000
        gr.Info(
            "You can only generate a dataset with 1000 or fewer rows. Setting to 1000."
        )
    if num_rows < 5:
        duration = 25
    elif num_rows < 10:
        duration = 60
    elif num_rows < 30:
        duration = 120
    elif num_rows < 100:
        duration = 240
    elif num_rows < 300:
        duration = 600
    elif num_rows < 1000:
        duration = 1200
    else:
        duration = 2400

    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_run_pipeline,
        args=(result_queue, num_turns, num_rows, system_prompt, is_sample),
    )

    try:
        p.start()
        total_steps = 100
        for step in range(total_steps):
            if not p.is_alive() or p._popen.poll() is not None:
                break
            progress(
                (step + 1) / total_steps,
                desc=f"Generating dataset with {num_rows} rows. Don't close this window.",
            )
            time.sleep(duration / total_steps)  # Adjust this value based on your needs
        p.join()
    except Exception as e:
        raise gr.Error(f"An error occurred during dataset generation: {str(e)}")

    distiset = result_queue.get()

    if repo_id is not None:
        progress(0.95, desc="Pushing dataset to Hugging Face Hub.")
        distiset.push_to_hub(
            repo_id=repo_id,
            private=private,
            include_script=False,
            token=oauth_token,
        )

    # If not pushing to hub generate the dataset directly
    distiset = distiset["default"]["train"]
    if num_turns == 1:
        outputs = distiset.to_pandas()[["prompt", "completion"]]
    else:
        outputs = distiset.to_pandas()[["messages"]]

    progress(1.0, desc="Dataset generation completed")
    return pd.DataFrame(outputs)


css = """
.main_ui_logged_out{opacity: 0.3; pointer-events: none}
"""

with gr.Blocks(
    title="⚗️ Distilabel Dataset Generator",
    head="⚗️ Distilabel Dataset Generator",
    css=css,
) as app:
    with gr.Row():
        gr.Markdown(
            "To push the dataset to the Hugging Face Hub you need to sign in. This will only be used for pushing the dataset not for data generation."
        )
    with gr.Row():
        gr.Column(scale=0.5)
        get_login_button()
        gr.Column(scale=0.5)

    gr.Markdown("## Iterate on a sample dataset")
    with gr.Column() as main_ui:
        dataset_description = gr.TextArea(
            label="Give a precise description of the assistant or tool. Don't describe the dataset",
            value=DEFAULT_DATASET_DESCRIPTIONS[0],
        )
        examples = gr.Examples(
            elem_id="system_prompt_examples",
            examples=[[example] for example in DEFAULT_DATASET_DESCRIPTIONS[1:]],
            inputs=[dataset_description],
        )
        with gr.Row():
            gr.Column(scale=1)
            btn_generate_system_prompt = gr.Button(value="Generate sample")
            gr.Column(scale=1)

        system_prompt = gr.TextArea(
            label="System prompt for dataset generation. You can tune it and regenerate the sample",
            value=DEFAULT_SYSTEM_PROMPT,
        )

        with gr.Row():
            table = gr.DataFrame(
                value=DEFAULT_DATASET,
                label="Sample dataset. Prompts and completions truncated to 256 tokens.",
                interactive=False,
                wrap=True,
            )

        with gr.Row():
            gr.Column(scale=1)
            btn_generate_sample_dataset = gr.Button(
                value="Regenerate sample",
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
            outputs=[table],
            show_progress=True,
        )

        btn_generate_sample_dataset.click(
            fn=generate_sample_dataset,
            inputs=[system_prompt],
            outputs=[table],
            show_progress=True,
        )

        # Add a header for the full dataset generation section
        gr.Markdown("## Generate full dataset")
        gr.Markdown(
            "Once you're satisfied with the sample, generate a larger dataset and push it to the Hub."
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

            with gr.Row(variant="panel"):
                oauth_token = gr.Textbox(
                    value=get_token(),
                    label="Hugging Face Token",
                    placeholder="hf_...",
                    type="password",
                    visible=False,
                )
                org_name = get_org_dropdown()
                repo_name = gr.Textbox(
                    label="Repo name", placeholder="dataset_name", value="my-distiset"
                )
                private = gr.Checkbox(
                    label="Private dataset", value=True, interactive=True, scale=0.5
                )
            with gr.Row() as regenerate_row:
                gr.Column(scale=1)
                btn_generate_full_dataset = gr.Button(
                    value="Generate Full Dataset", variant="primary"
                )
                gr.Column(scale=1)
            success_message = gr.Markdown(visible=False)
            with gr.Row():
                final_dataset = gr.DataFrame(
                    value=DEFAULT_DATASET,
                    label="Generated dataset",
                    interactive=False,
                    wrap=True,
                )

    def show_success_message(org_name, repo_name):
        return gr.Markdown(
            value=f"""
            <div style="padding: 1em; background-color: #e6f3e6; border-radius: 5px; margin-top: 1em;">
                <h3 style="color: #2e7d32; margin: 0;">Dataset Published Successfully!</h3>
                <p style="margin-top: 0.5em;">
                    The generated dataset is in the right format for Fine-tuning with TRL, AutoTrain or other frameworks.
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

    btn_generate_full_dataset.click(
        fn=hide_success_message,
        outputs=[success_message],
    ).then(
        fn=generate_dataset,
        inputs=[
            system_prompt,
            num_turns,
            num_rows,
            private,
            org_name,
            repo_name,
            oauth_token,
        ],
        outputs=[final_dataset],
        show_progress=True,
    ).success(
        fn=show_success_message,
        inputs=[org_name, repo_name],
        outputs=[success_message],
    )

    gr.Markdown("## Or run this pipeline locally with distilabel")

    with gr.Accordion("Run this pipeline on Distilabel", open=False):
        pipeline_code = gr.Code(
            value=generate_pipeline_code(
                system_prompt.value, num_turns.value, num_rows.value
            ),
            language="python",
            label="Distilabel Pipeline Code",
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
    app.load(get_token, outputs=[oauth_token])
    app.load(get_org_dropdown, outputs=[org_name])
    app.load(fn=swap_visibilty, outputs=main_ui)
