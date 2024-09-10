import multiprocessing

import gradio as gr
import pandas as pd
from distilabel.distiset import Distiset
from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns
from distilabel.steps.tasks import MagpieGenerator, TextGeneration

from src.distilabel_dataset_generator.utils import (
    OAuthToken,
    get_duplicate_button,
    get_login_button,
    get_org_dropdown,
    swap_visibilty,
)

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
DEFAULT_SYSTEM_PROMPT_DESCRIPTION = (
    "A chemistry dataset for an assistant that explains chemical reactions and formulas"
)
DEFAULT_SYSTEM_PROMPT = "You are an AI assistant specializing in chemistry and chemical reactions. Your purpose is to help users understand and work with chemical formulas, equations, and reactions. Provide clear explanations of reaction mechanisms, assist in balancing chemical equations, and offer guidance on the interpretation of chemical structures. Explain the roles of reactants, products, catalysts, and solvents, and define key chemistry terms when necessary."
DEFAULT_DATASET = pd.DataFrame(
    {
        "instruction": [
            "What is the term for the study of the structure and evolution of the Earth's interior.	"
        ],
        "response": [
            """The study of the structure and evolution of the Earth's interior is called geophysics, particularly the subfield of geology known as geodynamics, and more specifically the subfield of geology known as geotectonics. However, a more specific term for this study is "geology of the Earth's interior" or "Earth internal structure." However, the most commonly used term for this study is geophysics.	"""
        ],
    }
)


def _run_pipeline(result_queue, num_turns, num_rows, system_prompt, token: str = None):
    if num_turns == 1:
        output_mappings = {"instruction": "prompt", "response": "completion"}
    else:
        output_mappings = {"conversation": "messages"}
    with Pipeline(name="sft") as pipeline:
        magpie = MagpieGenerator(
            llm=InferenceEndpointsLLM(
                model_id=MODEL,
                tokenizer_id=MODEL,
                magpie_pre_query_template="llama3",
                generation_kwargs={
                    "temperature": 0.8,  # it's the best value for Llama 3.1 70B Instruct
                    "do_sample": True,
                    "max_new_tokens": 2048,
                    "stop_sequences": [
                        "<|eot_id|>",
                        "<|end_of_text|>",
                        "<|start_header_id|>",
                        "<|end_header_id|>",
                        "assistant",
                    ],
                },
                api_key=token,
            ),
            n_turns=num_turns,
            num_rows=num_rows,
            system_prompt=system_prompt,
            output_mappings=output_mappings,
        )
        keep_columns = KeepColumns(
            columns=list(output_mappings.values()) + ["model_name"],
        )
        magpie.connect(keep_columns)
    distiset: Distiset = pipeline.run(use_cache=False)
    result_queue.put(distiset)


def generate_system_prompt(dataset_description, token: OAuthToken = None):
    generate_description = TextGeneration(
        llm=InferenceEndpointsLLM(
            model_id=MODEL,
            tokenizer_id=MODEL,
            generation_kwargs={
                "temperature": 0.8,
                "max_new_tokens": 2048,
                "do_sample": True,
            },
        ),
        use_system_prompt=True,
    )
    generate_description.load()
    return next(
        generate_description.process(
            [
                {
                    "system_prompt": PROMPT_CREATION_PROMPT,
                    "instruction": dataset_description,
                }
            ]
        )
    )[0]["generation"]


def generate_dataset(
    system_prompt,
    num_turns=1,
    num_rows=5,
    private=True,
    orgs_selector=None,
    dataset_name=None,
    token: OAuthToken = None,
):
    if dataset_name is not None:
        if not dataset_name:
            raise gr.Error("Please provide a dataset name to push the dataset to.")
        if token is None:
            raise gr.Error(
                "Please sign in with Hugging Face to be able to push the dataset to the Hub."
            )
    if num_turns > 4:
        raise gr.Info(
            "You can only generate a dataset with 4 or fewer turns. Setting to 4."
        )
        num_turns = 4
    if num_rows > 5000:
        raise gr.Info(
            "You can only generate a dataset with 5000 or fewer rows. Setting to 5000."
        )
        num_rows = 5000

    gr.Info(
        "Started pipeline execution. This might take a while, depending on the number of rows and turns you have selected. Don't close this page."
    )
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_run_pipeline,
        args=(result_queue, num_turns, num_rows, system_prompt),
    )
    p.start()
    p.join()
    distiset = result_queue.get()

    if dataset_name is not None:
        gr.Info("Pushing dataset to Hugging Face Hub.")
        repo_id = f"{orgs_selector}/{dataset_name}"
        distiset.push_to_hub(
            repo_id=repo_id,
            private=private,
            include_script=False,
            token=token.token,
        )
        gr.Info(
            f'Dataset pushed to Hugging Face Hub: <a href="https://huggingface.co/datasets/{repo_id}">https://huggingface.co/datasets/{repo_id}</a>'
        )
    else:
        # If not pushing to hub generate the dataset directly
        distiset = distiset["default"]["train"]
        if num_turns == 1:
            outputs = distiset.to_pandas()[["prompt", "completion"]]
        else:
            outputs = {"conversation_id": [], "role": [], "content": []}
            conversations = distiset["messages"]
            for idx, entry in enumerate(conversations):
                for message in entry["messages"]:
                    outputs["conversation_id"].append(idx + 1)
                    outputs["role"].append(message["role"])
                    outputs["content"].append(message["content"])
        return pd.DataFrame(outputs)


with gr.Blocks(
    title="⚗️ Distilabel Dataset Generator",
    head="⚗️ Distilabel Dataset Generator",
) as app:
    gr.Markdown(
        """
### Generate a high quality SFT dataset in a breeze using [🐦‍⬛MagPie](https://arxiv.org/abs/2406.08464) and [🦙Llama 3.1 - 70B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct).

More information on distilabel and techniques can be found in the "FAQ" tab. The code can be found in the [Spaces repository](https://huggingface.co/spaces/argilla/distilabel-dataset-generator/tree/main).
"""
    )
    btn_duplicate = get_duplicate_button()

    dataset_description = gr.Textbox(
        label="Provide a description of the dataset",
        value=DEFAULT_SYSTEM_PROMPT_DESCRIPTION,
    )

    btn_generate_system_prompt = gr.Button(
        value="🧪 Generate Sytem Prompt and Sample Dataset"
    )

    system_prompt = gr.Textbox(
        label="Provide or correct the system prompt",
        value=DEFAULT_SYSTEM_PROMPT,
    )

    btn_generate_sample_dataset = gr.Button(
        value="🧪 Generate Sample Dataset of 5 rows and a single turn",
    )

    table = gr.Dataframe(label="Generated Dataset", wrap=True, value=DEFAULT_DATASET)

    btn_generate_system_prompt.click(
        fn=generate_system_prompt,
        inputs=[dataset_description],
        outputs=[system_prompt],
    ).then(
        fn=generate_dataset,
        inputs=[system_prompt],
        outputs=[table],
    )

    btn_generate_sample_dataset.click(
        fn=generate_dataset,
        inputs=[system_prompt],
        outputs=[table],
    )

    btn_login: gr.LoginButton | None = get_login_button()
    with gr.Column() as push_to_hub_ui:
        with gr.Row(variant="panel"):
            num_turns = gr.Number(
                value=1,
                label="Number of turns in the conversation",
                maximum=4,
                info="Whether the dataset is for a single turn with 'instruction-response' columns or a multi-turn conversation with a 'conversation' column.",
            )
            num_rows = gr.Number(
                value=100,
                label="Number of rows in the dataset",
                minimum=1,
                maximum=5000,
                info="The number of rows in the dataset. Note that you are able to generate more rows at once but that this will take time.",
            )
            private = gr.Checkbox(label="Private dataset", value=True, interactive=True)

        with gr.Row(variant="panel"):
            orgs_selector = gr.Dropdown(label="Organization")
            dataset_name_push_to_hub = gr.Textbox(label="Dataset Name to push to Hub")

        btn_generate_full_dataset = gr.Button(
            value="⚗️ Generate Full Dataset", variant="primary"
        )

        btn_generate_full_dataset.click(
            fn=generate_dataset,
            inputs=[
                system_prompt,
                num_turns,
                num_rows,
                private,
                orgs_selector,
                dataset_name_push_to_hub,
            ],
        )

    app.load(get_org_dropdown, outputs=[orgs_selector])
    app.load(fn=swap_visibilty, outputs=push_to_hub_ui)
