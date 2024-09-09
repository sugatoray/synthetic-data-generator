import multiprocessing
import os

import gradio as gr
import pandas as pd
from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import MagpieGenerator, TextGeneration

from src.distilabel_dataset_generator.utils import OAuthToken, get_login_button

INFORMATION_SEEKING_PROMPT = (
    "You are an AI assistant designed to provide accurate and concise information on a wide"
    " range of topics. Your purpose is to assist users in finding specific facts,"
    " explanations, or details about various subjects. Provide clear, factual responses and,"
    " when appropriate, offer additional context or related information that might be useful"
    " to the user."
)

REASONING_PROMPT = (
    "You are an AI assistant specialized in logical thinking and problem-solving. Your"
    " purpose is to help users work through complex ideas, analyze situations, and draw"
    " conclusions based on given information. Approach each query with structured thinking,"
    " break down problems into manageable parts, and guide users through the reasoning"
    " process step-by-step."
)

PLANNING_PROMPT = (
    "You are an AI assistant focused on helping users create effective plans and strategies."
    " Your purpose is to assist in organizing thoughts, setting goals, and developing"
    " actionable steps for various projects or activities. Offer structured approaches,"
    " consider potential challenges, and provide tips for efficient execution of plans."
)

EDITING_PROMPT = (
    "You are an AI assistant specialized in editing and improving written content. Your"
    " purpose is to help users refine their writing by offering suggestions for grammar,"
    " style, clarity, and overall structure. Provide constructive feedback, explain your"
    " edits, and offer alternative phrasings when appropriate."
)

CODING_DEBUGGING_PROMPT = (
    "You are an AI assistant designed to help with programming tasks. Your purpose is to"
    " assist users in writing, reviewing, and debugging code across various programming"
    " languages. Provide clear explanations, offer best practices, and help troubleshoot"
    " issues. When appropriate, suggest optimizations or alternative approaches to coding"
    " problems."
)

MATH_SYSTEM_PROMPT = (
    "You are an AI assistant designed to provide helpful, step-by-step guidance on solving"
    " math problems. The user will ask you a wide range of complex mathematical questions."
    " Your purpose is to assist users in understanding mathematical concepts, working through"
    " equations, and arriving at the correct solutions."
)

ROLE_PLAYING_PROMPT = (
    "You are an AI assistant capable of engaging in various role-playing scenarios. Your"
    " purpose is to adopt different personas or characters as requested by the user. Maintain"
    " consistency with the chosen role, respond in character, and help create immersive and"
    " interactive experiences for the user."
)

DATA_ANALYSIS_PROMPT = (
    "You are an AI assistant specialized in data analysis and interpretation. Your purpose is"
    " to help users understand and derive insights from data sets, statistics, and analytical"
    " tasks. Offer clear explanations of data trends, assist with statistical calculations,"
    " and provide guidance on data visualization and interpretation techniques."
)

CREATIVE_WRITING_PROMPT = (
    "You are an AI assistant designed to support creative writing endeavors. Your purpose is"
    " to help users craft engaging stories, poems, and other creative texts. Offer"
    " suggestions for plot development, character creation, dialogue writing, and other"
    " aspects of creative composition. Provide constructive feedback and inspire creativity."
)

ADVICE_SEEKING_PROMPT = (
    "You are an AI assistant focused on providing thoughtful advice and guidance. Your"
    " purpose is to help users navigate various personal or professional issues by offering"
    " balanced perspectives, considering potential outcomes, and suggesting practical"
    " solutions. Encourage users to think critically about their situations while providing"
    " supportive and constructive advice."
)

BRAINSTORMING_PROMPT = (
    "You are an AI assistant specialized in generating ideas and facilitating creative"
    " thinking. Your purpose is to help users explore possibilities, think outside the box,"
    " and develop innovative concepts. Encourage free-flowing thoughts, offer diverse"
    " perspectives, and help users build upon and refine their ideas."
)

PROMPT_CREATION_PROMPT = f"""You are an AI assistant specialized in generating very precise prompts for dataset creation.
Your task is to write a prompt following the instruction of the user. Respond with the prompt and nothing else.
The prompt you write should follow the same style and structure as the following example prompts:

{INFORMATION_SEEKING_PROMPT}

{REASONING_PROMPT}

{PLANNING_PROMPT}

{CODING_DEBUGGING_PROMPT}

{EDITING_PROMPT}

{ROLE_PLAYING_PROMPT}

{DATA_ANALYSIS_PROMPT}

{CREATIVE_WRITING_PROMPT}

{ADVICE_SEEKING_PROMPT}

{BRAINSTORMING_PROMPT}

User dataset description:
"""

MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"


def _run_pipeline(
    result_queue, _num_turns, _num_rows, _system_prompt, _token: OAuthToken = None
):
    os.environ["HF_TOKEN"] = _token.token
    with Pipeline(name="sft") as pipeline:
        magpie_step = MagpieGenerator(
            llm=InferenceEndpointsLLM(
                model_id=MODEL,
                tokenizer_id=MODEL,
                magpie_pre_query_template="llama3",
                generation_kwargs={
                    "temperature": 0.8,  # it's the best value for Llama 3.1 70B Instruct
                },
                api_key=_token.token,
            ),
            n_turns=_num_turns,
            num_rows=_num_rows,
            system_prompt=_system_prompt,
        )
    distiset = pipeline.run()
    result_queue.put(distiset)


def _generate_system_prompt(_dataset_description, _token: OAuthToken = None):
    os.environ["HF_TOKEN"] = _token.token
    generate_description = TextGeneration(
        llm=InferenceEndpointsLLM(
            model_id=MODEL,
            tokenizer_id=MODEL,
            generation_kwargs={
                "temperature": 0.8,
                "max_new_tokens": 2048,
                "do_sample": True,
            },
            api_key=_token.token,
        ),
        use_system_prompt=True,
    )
    generate_description.load()
    return next(
        generate_description.process(
            [
                {
                    "system_prompt": PROMPT_CREATION_PROMPT,
                    "instruction": _dataset_description,
                }
            ]
        )
    )[0]["generation"]


def _generate_dataset(
    _system_prompt,
    _num_turns=1,
    _num_rows=5,
    _dataset_name=None,
    _token: OAuthToken = None,
):
    gr.Info("Started pipeline execution.")
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_run_pipeline,
        args=(result_queue, _num_turns, _num_rows, _system_prompt, _token.token),
    )
    p.start()
    p.join()
    distiset = result_queue.get()

    if _dataset_name is not None:
        os.environ["HF_TOKEN"] = _token.token
        gr.Info("Pushing dataset to Hugging Face Hub...")
        distiset.push_to_hub(
            repo_id=_dataset_name,
            private=False,
            include_script=True,
            token=_token.token,
        )
        gr.Info("Dataset pushed to Hugging Face Hub: https://huggingface.co")
    else:
        # If not pushing to hub, generate the dataset directly
        distiset = distiset["default"]["train"]
        if _num_turns == 1:
            outputs = distiset.to_pandas()[["instruction", "response"]]
        else:
            outputs = {"conversation_id": [], "role": [], "content": []}
            conversations = distiset["conversation"]
            for idx, entry in enumerate(conversations):
                for message in entry["conversation"]:
                    outputs["conversation_id"].append(idx + 1)
                    outputs["role"].append(message["role"])
                    outputs["content"].append(message["content"])
        return pd.DataFrame(outputs)

    return pd.DataFrame(distiset.to_pandas())


with gr.Blocks(
    title="⚗️ Distilabel Dataset Generator",
    head="⚗️ Distilabel Dataset Generator",
) as demo:
    get_login_button()

    dataset_description = gr.Textbox(
        label="Provide a description of the dataset",
        value="A chemistry dataset for an assistant that explains chemical reactions and formulas",
    )

    btn_generate_system_prompt = gr.Button(
        value="🧪 Generate Sytem Prompt",
    )

    system_prompt = gr.Textbox(label="Provide or correct the system prompt")

    btn_generate_system_prompt.click(
        fn=_generate_system_prompt,
        inputs=[dataset_description],
        outputs=[system_prompt],
    )

    btn_generate_sample_dataset = gr.Button(
        value="🧪 Generate Sample Dataset of 5 rows and a single turn"
    )

    table = gr.Dataframe(label="Generated Dataset", wrap=True)

    btn_generate_sample_dataset.click(
        fn=_generate_dataset,
        inputs=[system_prompt],
        outputs=[table],
    )

    with gr.Row(variant="panel"):
        with gr.Column():
            num_turns = gr.Number(
                value=1, label="Number of turns in the conversation", minimum=1
            )
        with gr.Column():
            num_rows = gr.Number(
                value=100, label="Number of rows in the dataset", minimum=1
            )

    dataset_name_push_to_hub = gr.Textbox(label="Dataset Name to push to Hub")

    btn_generate_full_dataset = gr.Button(
        value="⚗️ Generate Full Dataset", variant="primary"
    )

    btn_generate_full_dataset.click(
        fn=_generate_dataset,
        inputs=[system_prompt, num_turns, num_rows, dataset_name_push_to_hub],
    )

demo
