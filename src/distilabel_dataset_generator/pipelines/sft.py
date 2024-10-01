import pandas as pd
from datasets import Dataset
from distilabel.distiset import Distiset
from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns
from distilabel.steps.tasks import ChatGeneration, MagpieGenerator, TextGeneration

from src.distilabel_dataset_generator.utils import HF_TOKENS

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

In the generated prompt always finish with this sentence: User questions are direct and concise.

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

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_DATASET_DESCRIPTIONS = (
    "rude customer assistant for a phone company",
    "assistant that solves math puzzles using python",
)
DEFAULT_SYSTEM_PROMPTS = [
    """You are a customer support agent for a phone company. Your purpose is to assist customers with their phone-related issues, but you are not very patient and tend to be a bit rude. User queries will be straightforward and clear, but you will respond in a somewhat blunt and curt manner. Remember to keep your responses concise and to the point. User queries are often about phone plans, billing, and technical issues. Your responses should be direct and focus on resolving the issue at hand, but with a slightly abrasive tone. User queries will be concise and to the point, User queries are often about phone plans, billing, and technical issues.""",
    """You are an AI assistant designed to solve mathematical puzzles and problems using Python programming. Your purpose is to help users tackle various math-related challenges by writing, testing, and explaining Python code. Provide step-by-step solutions, break down complex problems into manageable parts, and offer clear explanations of mathematical concepts and their implementation in Python. Assist users in understanding the logic behind the code and the mathematical principles that govern the solution. User questions are direct and concise.""",
]
DEFAULT_DATASETS = [
    pd.DataFrame.from_dict(
        {
            "prompt": [
                "I'm having trouble with my phone signal at home. It's weak. What can you do to help me?"
            ],
            "completion": [
                """Look, we're not responsible for your signal quality at your house. It's probably a homeowner issue, like your roof covering our tower or something. Can you check if you have a clear view of the roof or if there are any obstructions?"""
            ],
        }
    ),
    pd.DataFrame.from_dict(
        {
            "prompt": [
                "If the product of the largest and smallest numbers of the five given numbers (3, 7, 8, 12, 20) is greater than the sum of the other three numbers, then the largest number among them is A) 7 B) 12 C) 20 D) 8 Why is this statement true?"
            ],
            "completion": [
                """**To solve this problem, we will follow these steps: 1. Calculate the product of the largest and smallest numbers among the given set. 2. Calculate the sum of the remaining three numbers. 3. Compare the product with the sum to check the given condition. Here's the Python code to solve the problem: ```python # Define the given numbers numbers = [3, 7, 8, 12, 20] # Sort the numbers in ascending order numbers.sort() # Get the smallest number (first element after sorting) and the largest number (last element after sorting) smallest_number = numbers[0] largest_number = numbers[-1] # Calculate the product of the smallest and largest numbers product = smallest_number * largest_number # Calculate the sum of the remaining three numbers sum_of_remaining_numbers = sum(numbers[1:-1]) # Check if the product is greater than the sum of the remaining numbers if product > sum_of_remaining_numbers: print("The statement is true.") else: print("The statement is false.") # Print the largest number among the options largest_option = [7, 12, 20] print("The largest number among the options is:", max(largest"""
            ],
        }
    ),
]
_STOP_SEQUENCES = [
    "<|eot_id|>",
    "<|start_header_id|>",
    "assistant",
    " \n\n",
]
DEFAULT_BATCH_SIZE = 5
TOKEN_INDEX = 0


def _get_output_mappings(num_turns):
    if num_turns == 1:
        return {"instruction": "prompt", "response": "completion"}
    else:
        return {"conversation": "messages"}


def generate_pipeline_code(system_prompt, num_turns, num_rows):
    input_mappings = _get_output_mappings(num_turns)
    code = f"""
# Requirements: `pip install distilabel[hf-inference-endpoints]`
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns
from distilabel.steps.tasks import MagpieGenerator
from distilabel.llms import InferenceEndpointsLLM

MODEL = "{MODEL}"
SYSTEM_PROMPT = "{system_prompt}"

with Pipeline(name="sft") as pipeline:
    magpie = MagpieGenerator(
        llm=InferenceEndpointsLLM(
            model_id=MODEL,
            tokenizer_id=MODEL,
            magpie_pre_query_template="llama3",
            generation_kwargs={{
                "temperature": 0.8,
                "do_sample": True,
                "max_new_tokens": 2048,
                "stop_sequences": {_STOP_SEQUENCES}
            }}
        ),
        n_turns={num_turns},
        num_rows={num_rows},
        batch_size=1,
        system_prompt=SYSTEM_PROMPT,
        output_mappings={input_mappings},
    )
    keep_columns = KeepColumns(
        columns={list(input_mappings.values())} + ["model_name", "system_prompt"],
    )
    magpie.connect(keep_columns)

if __name__ == "__main__":
    distiset = pipeline.run()
"""
    return code


def _get_next_api_key():
    global TOKEN_INDEX
    api_key = HF_TOKENS[TOKEN_INDEX % len(HF_TOKENS)]
    TOKEN_INDEX += 1
    return api_key


def get_magpie_generator(num_turns, num_rows, system_prompt, is_sample):
    input_mappings = _get_output_mappings(num_turns)
    output_mappings = input_mappings.copy()
    if num_turns == 1:
        magpie_generator = MagpieGenerator(
            llm=InferenceEndpointsLLM(
                model_id=MODEL,
                tokenizer_id=MODEL,
                api_key=_get_next_api_key(),
                magpie_pre_query_template="llama3",
                generation_kwargs={
                    "temperature": 0.8,
                    "do_sample": True,
                    "max_new_tokens": 256 if is_sample else 512,
                    "stop_sequences": _STOP_SEQUENCES,
                },
            ),
            batch_size=DEFAULT_BATCH_SIZE,
            n_turns=num_turns,
            num_rows=num_rows,
            system_prompt=system_prompt,
            output_mappings=output_mappings,
            only_instruction=True,
        )
    else:
        magpie_generator = MagpieGenerator(
            llm=InferenceEndpointsLLM(
                model_id=MODEL,
                tokenizer_id=MODEL,
                api_key=_get_next_api_key(),
                magpie_pre_query_template="llama3",
                generation_kwargs={
                    "temperature": 0.8,
                    "do_sample": True,
                    "max_new_tokens": 256 if is_sample else 1024,
                    "stop_sequences": _STOP_SEQUENCES,
                },
            ),
            batch_size=DEFAULT_BATCH_SIZE,
            end_with_user=True,
            n_turns=num_turns,
            num_rows=num_rows,
            system_prompt=system_prompt,
            output_mappings=output_mappings,
        )
    magpie_generator.load()
    return magpie_generator


def get_response_generator(num_turns, system_prompt, is_sample):
    if num_turns == 1:
        response_generator = TextGeneration(
            llm=InferenceEndpointsLLM(
                model_id=MODEL,
                tokenizer_id=MODEL,
                api_key=_get_next_api_key(),
                generation_kwargs={
                    "temperature": 0.8,
                    "max_new_tokens": 256 if is_sample else 1024,
                },
            ),
            system_prompt=system_prompt,
            output_mappings={"generation": "completion"},
            input_mappings={"instruction": "prompt"},
        )
    else:
        response_generator = ChatGeneration(
            llm=InferenceEndpointsLLM(
                model_id=MODEL,
                tokenizer_id=MODEL,
                api_key=_get_next_api_key(),
                generation_kwargs={
                    "temperature": 0.8,
                    "max_new_tokens": 2048,
                },
            ),
            output_mappings={"generation": "completion"},
            input_mappings={"conversation": "messages"},
        )
    response_generator.load()
    return response_generator


def get_prompt_generator():
    global TOKEN_INDEX
    api_key = HF_TOKENS[TOKEN_INDEX % len(HF_TOKENS)]
    TOKEN_INDEX += 1
    prompt_generator = TextGeneration(
        llm=InferenceEndpointsLLM(
            api_key=api_key,
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
    prompt_generator.load()
    return prompt_generator


def get_pipeline(num_turns, num_rows, system_prompt, is_sample):
    input_mappings = _get_output_mappings(num_turns)
    output_mappings = input_mappings

    with Pipeline(name="sft") as pipeline:
        magpie = get_magpie_generator(num_turns, num_rows, system_prompt, is_sample)
        generate_response = get_response_generator(system_prompt, is_sample)

        keep_columns = KeepColumns(
            columns=list(output_mappings.values()) + ["model_name"],
        )

        magpie.connect(generate_response)
        generate_response.connect(keep_columns)
        return pipeline


if __name__ == "__main__":
    prompt_generation_step = get_prompt_generator()
    system_prompt = next(
        prompt_generation_step.process(
            [
                {
                    "system_prompt": PROMPT_CREATION_PROMPT,
                    "instruction": DEFAULT_DATASET_DESCRIPTIONS[0],
                }
            ]
        )
    )[0]["generation"]
    num_rows = 2
    num_turns = 1
    magpie_generator = get_magpie_generator(num_turns, num_rows, system_prompt, False)
    response_generator = get_response_generator(num_turns, system_prompt, False)
    total_steps = num_rows * 2
    batch_size = 5  # Adjust this value as needed

    # create instructions
    magpie_results = []
    for i in range(0, num_rows, batch_size):
        batch = list(magpie_generator.process())[:batch_size]
        magpie_results.extend([item[0] for item in batch])

    # generate responses
    response_results = []
    if num_turns == 1:
        for i in range(0, len(magpie_results), batch_size):
            batch = magpie_results[i : i + batch_size]
            batch = [entry[0] for entry in batch]
            responses = list(response_generator.process(inputs=batch))
            response_results.extend(responses)
        for result in response_results:
            result[0]["prompt"] = result[0]["instruction"]
            result[0]["completion"] = result[0]["generation"]
            result[0]["system_prompt"] = system_prompt
    else:
        for result in magpie_results:
            result[0]["conversation"].insert(
                0, {"role": "system", "content": system_prompt}
            )
            result[0]["messages"] = result[0]["conversation"]
        for i in range(0, len(magpie_results), batch_size):
            batch = magpie_results[i : i + batch_size]
            batch = [entry[0] for entry in batch]
            responses = list(response_generator.process(inputs=batch))
            response_results.extend(responses)

        for result in response_results:
            result[0]["messages"].append(
                {"role": "assistant", "content": result[0]["generation"]}
            )

    distiset_results = []
    for result in response_results[0]:
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
