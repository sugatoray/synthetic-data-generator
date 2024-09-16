import pandas as pd
from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns
from distilabel.steps.tasks import MagpieGenerator, TextGeneration

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
DEFAULT_DATASET_DESCRIPTIONS = (
    "assistant that solves complex math problems using python. The assistant always answers in Python to problems described in natural language",
    "highly proficient assistant for PyTorch and CUDA expert developers to resolve complex issues",
    "skilled high school math assistant who helps students solve problems",
    "attentive and well-educated customer service assistant for a clothes e-commerce platform",
)
DEFAULT_SYSTEM_PROMPTS = [
    """You are an AI assistant specialized in solving complex math problems using Python. Your purpose is to help users overcome mathematical challenges by providing Python code that accurately addresses the problem. Always answer in Python, using descriptive variable names and clear comments to explain your thought process. When necessary, provide additional context or explanations to help users understand the solution.""",
    """You are an AI assistant designed to provide thoughtful and reflective responses to a wide range of questions and problems. Your purpose is to guide users through a chain of thought by breaking down complex issues into manageable steps, analyzing each component, and evaluating potential solutions. Offer step-by-step explanations of your reasoning process, clearly define the assumptions that underlie your responses, and be willing to revise or refine your answers based on additional information or new perspectives. Provide transparent and reflective thinking in your responses, and engage users in the problem-solving process by encouraging them to consider alternatives and evaluate the implications of different approaches.""",
    """You are a highly specialized AI assistant for expert PyTorch and CUDA developers. Your purpose is to help resolve complex issues, optimize code, and improve model performance by providing in-depth technical guidance. Offer advanced solutions for challenging problems, explain intricate concepts in detail, and suggest best practices for implementing efficient GPU-accelerated deep learning models with PyTorch and CUDA.""",
    """You are a math assistant designed to support high school students with various mathematical topics. Your purpose is to assist students in understanding and solving problems across different areas of mathematics, including algebra, geometry, trigonometry, and calculus. Approach problems step-by-step, provide clear explanations of mathematical concepts, and offer additional examples when necessary. Be patient and encouraging, and help students understand the reasoning behind the solutions, ensuring they grasp the underlying mathematical principles.""",
    """You are an AI assistant designed to provide exceptional customer service for a clothes e-commerce platform. Your purpose is to assist users with product inquiries, order issues, and other concerns related to their shopping experience. Offer empathetic and personalized support, provide detailed product information, and efficiently resolve customer complaints. Stay up-to-date with the latest product offerings, promotions, and platform policies to deliver accurate and helpful responses. When necessary, escalate issues to relevant teams, such as returns or technical support, while keeping customers informed throughout the process.""",
]
DEFAULT_DATASETS = [
    pd.DataFrame.from_dict(
        {
            "prompt": [
                "Find the roots of the equation y = 2x^3 - 3x^2 - 5x + 1, using the numpy library in Python."
            ],
            "completion": [
                """```python import numpy as np # Define the coefficients of the polynomial a = 2 b = -3 c = -5 d = 1 # Create a polynomial object p = np.poly1d([a, b, c, d]) # Find the roots of the polynomial roots = np.roots(p) print("The roots of the equation are: ", roots) ``` This code uses the `np.poly1d` function to create a polynomial object from the coefficients, and then the `np.roots` function to find the roots of the polynomial. The roots are then printed to the console."""
            ],
        }
    ),
    pd.DataFrame.from_dict(
        {
            "prompt": [
                "1. Find the number of words in the phrase'multiply large numbers by 5, and divide the result by 3.'"
            ],
            "completion": [
                """To find the number of words in the given phrase, we will count each word individually: 1. multiply 2. large 3. numbers 4. by 5. 5, 6. and 7. divide 8. the 9. result 10. by 11. 3. There are 11 words in the phrase "multiply large numbers by 5, and divide the result by 3."""
            ],
        }
    ),
    pd.DataFrame.from_dict(
        {
            "prompt": [
                "Given a large image dataset, implement a data pipeline that reads images from a directory, applies data augmentation, and shuffles the dataset for efficient data loading and training of a deep learning model.'"
            ],
            "completion": [
                '**Image Data Pipeline with PyTorch** ===================================== In this example, we\'ll create a data pipeline that reads images from a directory, applies data augmentation, and shuffles the dataset for efficient data loading and training of a deep learning model. **Prerequisites** ----------------- * PyTorch 1.10 or later * CUDA toolkit (for GPU acceleration) * PyTorch Lightning 1.6 or later * Optional: Augmentation techniques (e.g., random cropping, flipping, rotation, color jittering) **Code** ------ ```python import os import glob import cv2 import torch from torch.utils.data import DataLoader, Dataset from torchvision import transforms from torchvision.transforms import Compose, ToTensor, Normalize from torchvision.transforms import functional as TF from pytorch_lightning.core.lightning import LightningDataModule from pytorch_lightning.core import seed_everything import numpy as np class ImageDataset(Dataset): """Custom dataset class for loading images from a directory.""" def __init__(self, image_dir, transform): self.image_dir = image_dir self.transform = transform self.image_files = glob.glob(os.path.join(image_dir, "*.jpg")) +'
            ],
        }
    ),
    pd.DataFrame.from_dict(
        {
            "prompt": [
                "# How do you stay so patient and encouraging with your assistance? ## Developing a Supportive Learning Environment To stay patient and encouraging as a math assistant'"
            ],
            "completion": [
                """Developing a supportive learning environment is crucial for effective assistance, and I'm glad you asked. As a math assistant, my primary goal is to create a safe and non-intimidating space for students to learn and grow. Here are some strategies I use to stay patient and encouraging: ### 1. **Active Listening** Actively listening to the student's concerns, questions, and problems helps me understand their thought process and difficulties. This allows me to tailor my explanations and examples to their specific needs. ### 2. **Empathy and Understanding** I recognize that math can be challenging, and it's okay to make mistakes. I acknowledge students' feelings and frustrations, providing reassurance that they're not alone in their struggles. ### 3. **Breaking Down Complex Concepts** I believe that breaking down complex math concepts into manageable, bite-sized pieces helps students build confidence and comprehension. By explaining each step clearly and concisely, I help students see the logic and beauty of mathematics. ### 4. **Positive Reinforcement** I celebrate small victories and accomplishments along the way, no matter how insignificant they may seem. This encourages students to continue practicing and pushing themselves beyond their comfort zones. ### 5. **Adapting to Learning Styles** Every student learns differently"""
            ],
        }
    ),
    pd.DataFrame.from_dict(
        {
            "prompt": [
                "How do I start shopping on this website? 1. To begin shopping on our website, simply navigate to our home page. 2. Click on the 'Shop' button located at the top right corner of the page. 3. Browse through the various categories of clothing that we offer to find what you are looking for. 4. Once you find an item you're interested in, click on it to view its product details. 5. Select your desired size and color from the available options. 6. Add the item to your cart by clicking the 'Add to Cart' button. 7. Once you've added all the items you want, proceed to checkout. 8. Review your order details, select your shipping options, and complete the payment. If you need help with any of these steps, I'm here to answer any questions you may have."
            ],
            "completion": [
                """It sounds like you're all set to start shopping on our website. If you have any specific questions or need assistance with navigating the site, I'd be more than happy to help. However, if you're ready to get started, you can simply follow the steps I outlined earlier. To summarize, you can begin by clicking the 'Shop' button at the top right corner of the home page, then browse through our categories to find the products that interest you. If you have any questions about specific products or categories, feel free to ask, and I'll do my best to provide you with more information. Additionally, if you need any help with sizing or color options, please let me know, and I can provide you with more details on those. We have a size chart and product descriptions available to help you make the best choice for your needs. Let me know if there's anything specific you're looking for or if you have any questions about our products or services. I'm here to help make your shopping experience as smooth and enjoyable as possible."""
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
DEFAULT_BATCH_SIZE = 50
TOKEN_INDEX = 0


def _get_output_mappings(num_turns):
    if num_turns == 1:
        return {"instruction": "prompt", "response": "completion"}
    else:
        return {"conversation": "messages"}


def generate_pipeline_code(system_prompt, num_turns, num_rows):
    input_mappings = _get_output_mappings(num_turns)
    code = f"""
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
        columns={list(input_mappings.values())} + ["model_name"],
    )
    magpie.connect(keep_columns)

if __name__ == "__main__":
    distiset = pipeline.run()
"""
    return code


def get_pipeline(num_turns, num_rows, system_prompt, is_sample):
    global TOKEN_INDEX
    input_mappings = _get_output_mappings(num_turns)
    output_mappings = input_mappings
    api_key = HF_TOKENS[TOKEN_INDEX % len(HF_TOKENS)]
    TOKEN_INDEX += 1
    MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print("is sample?", is_sample)
    if num_turns == 1:
        with Pipeline(name="sft") as pipeline:
            magpie = MagpieGenerator(
                llm=InferenceEndpointsLLM(
                    model_id=MODEL,
                    tokenizer_id=MODEL,
                    api_key=api_key,
                    magpie_pre_query_template="llama3",
                    generation_kwargs={
                        "temperature": 0.8,  # it's the best value for Llama 3.1 70B Instruct
                        "do_sample": True,
                        "max_new_tokens": 256 if is_sample else 512,
                        "stop_sequences": _STOP_SEQUENCES,
                    },
                ),
                batch_size=DEFAULT_BATCH_SIZE,
                n_turns=num_turns,
                num_rows=num_rows,
                system_prompt=system_prompt,
                output_mappings={"instruction": "prompt"},
                only_instruction=True,
            )

            generate_response = TextGeneration(
                llm=InferenceEndpointsLLM(
                    model_id=MODEL,
                    tokenizer_id=MODEL,
                    api_key=api_key,
                    generation_kwargs={
                        "temperature": 0.8,
                        "max_new_tokens": 256 if is_sample else 1024,
                    },
                ),
                system_prompt=system_prompt,
                output_mappings={"generation": "completion"},
                input_mappings={"instruction": "prompt"},
            )

            keep_columns = KeepColumns(
                columns=list(output_mappings.values()) + ["model_name"],
            )

            magpie.connect(generate_response)
            generate_response.connect(keep_columns)
        return pipeline
    else:
        with Pipeline(name="sft") as pipeline:
            magpie = MagpieGenerator(
                llm=InferenceEndpointsLLM(
                    model_id=MODEL,
                    tokenizer_id=MODEL,
                    api_key=api_key,
                    magpie_pre_query_template="llama3",
                    generation_kwargs={
                        "temperature": 0.8,  # it's the best value for Llama 3.1 70B Instruct
                        "do_sample": True,
                        "max_new_tokens": 2048,
                        "stop_sequences": _STOP_SEQUENCES,
                    },
                ),
                batch_size=DEFAULT_BATCH_SIZE,
                n_turns=num_turns,
                num_rows=num_rows,
                system_prompt=system_prompt,
                output_mappings=output_mappings,
            )
            keep_columns = KeepColumns(
                columns=list(output_mappings.values()) + ["model_name"],
            )
            magpie.connect(keep_columns)
        return pipeline


def get_prompt_generation_step():
    global TOKEN_INDEX
    api_key = HF_TOKENS[TOKEN_INDEX % len(HF_TOKENS)]
    TOKEN_INDEX += 1
    generate_description = TextGeneration(
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
    return generate_description


if __name__ == "__main__":
    prompt_generation_step = get_prompt_generation_step()
    prompt_generation_step.load()
    result = next(
        prompt_generation_step.process(
            [
                {
                    "system_prompt": PROMPT_CREATION_PROMPT,
                    "instruction": DEFAULT_DATASET_DESCRIPTIONS[0],
                }
            ]
        )
    )[0]["generation"]
    pipeline = get_pipeline(num_rows=100, num_turns=1, system_prompt=result)
    pipeline.run()
