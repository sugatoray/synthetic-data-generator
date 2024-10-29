import ast
from typing import Dict, List, Union

import argilla as rg
import gradio as gr
import pandas as pd
from datasets import Dataset
from distilabel.distiset import Distiset
from huggingface_hub import HfApi

from src.distilabel_dataset_generator.apps.base import (
    get_argilla_client,
    get_main_ui,
    get_pipeline_code_ui,
    hide_success_message,
    push_pipeline_code_to_hub,
    show_success_message_argilla,
    show_success_message_hub,
    validate_argilla_user_workspace_dataset,
)
from src.distilabel_dataset_generator.apps.base import (
    push_dataset_to_hub as push_to_hub_base,
)
from src.distilabel_dataset_generator.pipelines.base import (
    DEFAULT_BATCH_SIZE,
)
from src.distilabel_dataset_generator.pipelines.embeddings import (
    get_embeddings,
    get_sentence_embedding_dimensions,
)
from src.distilabel_dataset_generator.pipelines.sft import (
    DEFAULT_DATASET_DESCRIPTIONS,
    DEFAULT_DATASETS,
    DEFAULT_SYSTEM_PROMPTS,
    PROMPT_CREATION_PROMPT,
    generate_pipeline_code,
    get_magpie_generator,
    get_prompt_generator,
    get_response_generator,
)

TASK = "supervised_fine_tuning"


def convert_dataframe_messages(dataframe: pd.DataFrame) -> pd.DataFrame:
    def convert_to_list_of_dicts(messages: str) -> List[Dict[str, str]]:
        return ast.literal_eval(
            messages.replace("'user'}", "'user'},")
            .replace("'system'}", "'system'},")
            .replace("'assistant'}", "'assistant'},")
        )

    if "messages" in dataframe.columns:
        dataframe["messages"] = dataframe["messages"].apply(
            lambda x: convert_to_list_of_dicts(x) if isinstance(x, str) else x
        )
    return dataframe


def push_dataset_to_hub(
    dataframe: pd.DataFrame,
    private: bool = True,
    org_name: str = None,
    repo_name: str = None,
    oauth_token: Union[gr.OAuthToken, None] = None,
    progress=gr.Progress(),
):
    original_dataframe = dataframe.copy(deep=True)
    dataframe = convert_dataframe_messages(dataframe)
    try:
        push_to_hub_base(
            dataframe, private, org_name, repo_name, oauth_token, progress, task=TASK
        )
    except Exception as e:
        raise gr.Error(f"Error pushing dataset to the Hub: {e}")
    return original_dataframe


def push_dataset_to_argilla(
    dataframe: pd.DataFrame,
    dataset_name: str,
    oauth_token: Union[gr.OAuthToken, None] = None,
    progress=gr.Progress(),
) -> pd.DataFrame:
    original_dataframe = dataframe.copy(deep=True)
    dataframe = convert_dataframe_messages(dataframe)
    try:
        progress(0.1, desc="Setting up user and workspace")
        client = get_argilla_client()
        hf_user = HfApi().whoami(token=oauth_token.token)["name"]
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
        rg_dataset = client.datasets(name=dataset_name, workspace=hf_user)
        if rg_dataset is None:
            rg_dataset = rg.Dataset(
                name=dataset_name,
                workspace=hf_user,
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


def generate_system_prompt(dataset_description, progress=gr.Progress()):
    progress(0.0, desc="Generating system prompt")
    if dataset_description in DEFAULT_DATASET_DESCRIPTIONS:
        index = DEFAULT_DATASET_DESCRIPTIONS.index(dataset_description)
        if index < len(DEFAULT_SYSTEM_PROMPTS):
            return DEFAULT_SYSTEM_PROMPTS[index]

    progress(0.3, desc="Initializing text generation")
    generate_description = get_prompt_generator()
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
        desc="(2/2) Creating dataset",
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


(
    app,
    main_ui,
    custom_input_ui,
    dataset_description,
    examples,
    btn_generate_system_prompt,
    system_prompt,
    sample_dataset,
    btn_generate_sample_dataset,
    dataset_name,
    add_to_existing_dataset,
    btn_generate_full_dataset_argilla,
    btn_generate_and_push_to_argilla,
    btn_push_to_argilla,
    org_name,
    repo_name,
    private,
    btn_generate_full_dataset,
    btn_generate_and_push_to_hub,
    btn_push_to_hub,
    final_dataset,
    success_message,
) = get_main_ui(
    default_dataset_descriptions=DEFAULT_DATASET_DESCRIPTIONS,
    default_system_prompts=DEFAULT_SYSTEM_PROMPTS,
    default_datasets=DEFAULT_DATASETS,
    fn_generate_system_prompt=generate_system_prompt,
    fn_generate_dataset=generate_dataset,
    task=TASK,
)

with app:
    with main_ui:
        with custom_input_ui:
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

        pipeline_code = get_pipeline_code_ui(
            generate_pipeline_code(system_prompt.value, num_turns.value, num_rows.value)
        )

    # define app triggers
    gr.on(
        triggers=[
            btn_generate_full_dataset.click,
            btn_generate_full_dataset_argilla.click,
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
        fn=validate_argilla_user_workspace_dataset,
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
        fn=push_dataset_to_argilla,
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
        fn=push_dataset_to_hub,
        inputs=[final_dataset, private, org_name, repo_name],
        outputs=[final_dataset],
        show_progress=True,
    ).then(
        fn=push_pipeline_code_to_hub,
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
        fn=push_dataset_to_hub,
        inputs=[final_dataset, private, org_name, repo_name],
        outputs=[final_dataset],
        show_progress=True,
    ).then(
        fn=push_pipeline_code_to_hub,
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
        fn=validate_argilla_user_workspace_dataset,
        inputs=[dataset_name, final_dataset, add_to_existing_dataset],
        outputs=[final_dataset],
        show_progress=True,
    ).success(
        fn=push_dataset_to_argilla,
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
