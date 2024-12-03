import os
import warnings
from typing import Optional

import argilla as rg
import distilabel
import distilabel.distiset
from distilabel.utils.card.dataset_card import (
    DistilabelDatasetCard,
    size_categories_parser,
)
from huggingface_hub import DatasetCardData, HfApi

HF_TOKENS = [os.getenv("HF_TOKEN")] + [os.getenv(f"HF_TOKEN_{i}") for i in range(1, 10)]
HF_TOKENS = [token for token in HF_TOKENS if token]

if len(HF_TOKENS) == 0:
    raise ValueError(
        "HF_TOKEN is not set. Ensure you have set the HF_TOKEN environment variable that has access to the Hugging Face Hub repositories and Inference Endpoints."
    )

ARGILLA_API_URL = os.getenv("ARGILLA_API_URL")
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY")
if ARGILLA_API_URL is None or ARGILLA_API_KEY is None:
    ARGILLA_API_URL = os.getenv("ARGILLA_API_URL_SDG_REVIEWER")
    ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY_SDG_REVIEWER")

if ARGILLA_API_URL is None or ARGILLA_API_KEY is None:
    warnings.warn("ARGILLA_API_URL or ARGILLA_API_KEY is not set")
    argilla_client = None
else:
    argilla_client = rg.Argilla(
        api_url=ARGILLA_API_URL,
        api_key=ARGILLA_API_KEY,
    )


class CustomDistisetWithAdditionalTag(distilabel.distiset.Distiset):
    def _generate_card(
        self,
        repo_id: str,
        token: str,
        include_script: bool = False,
        filename_py: Optional[str] = None,
    ) -> None:
        """Generates a dataset card and pushes it to the Hugging Face Hub, and
        if the `pipeline.yaml` path is available in the `Distiset`, uploads that
        to the same repository.

        Args:
            repo_id: The ID of the repository to push to, from the `push_to_hub` method.
            token: The token to authenticate with the Hugging Face Hub, from the `push_to_hub` method.
            include_script: Whether to upload the script to the hugging face repository.
            filename_py: The name of the script. If `include_script` is True, the script will
                be uploaded to the repository using this name, otherwise it won't be used.
        """
        card = self._get_card(
            repo_id=repo_id,
            token=token,
            include_script=include_script,
            filename_py=filename_py,
        )

        card.push_to_hub(
            repo_id,
            repo_type="dataset",
            token=token,
        )
        if self.pipeline_path:
            # If the pipeline.yaml is available, upload it to the Hugging Face Hub as well.
            HfApi().upload_file(
                path_or_fileobj=self.pipeline_path,
                path_in_repo=distilabel.distiset.PIPELINE_CONFIG_FILENAME,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )

    def _get_card(
        self,
        repo_id: str,
        token: Optional[str] = None,
        include_script: bool = False,
        filename_py: Optional[str] = None,
    ) -> DistilabelDatasetCard:
        """Generates the dataset card for the `Distiset`.

        Note:
            If `repo_id` and `token` are provided, it will extract the metadata from the README.md file
            on the hub.

        Args:
            repo_id: Name of the repository to push to, or the path for the distiset if saved to disk.
            token: The token to authenticate with the Hugging Face Hub.
                We assume that if it's provided, the dataset will be in the Hugging Face Hub,
                so the README metadata will be extracted from there.
            include_script: Whether to upload the script to the hugging face repository.
            filename_py: The name of the script. If `include_script` is True, the script will
                be uploaded to the repository using this name, otherwise it won't be used.

        Returns:
            The dataset card for the `Distiset`.
        """
        sample_records = {}
        for name, dataset in self.items():
            sample_records[name] = (
                dataset[0] if not isinstance(dataset, dict) else dataset["train"][0]
            )

        readme_metadata = {}
        if repo_id and token:
            readme_metadata = self._extract_readme_metadata(repo_id, token)

        metadata = {
            **readme_metadata,
            "size_categories": size_categories_parser(
                max(len(dataset) for dataset in self.values())
            ),
            "tags": [
                "synthetic",
                "distilabel",
                "rlaif",
                "datacraft",
            ],
        }

        card = DistilabelDatasetCard.from_template(
            card_data=DatasetCardData(**metadata),
            repo_id=repo_id,
            sample_records=sample_records,
            include_script=include_script,
            filename_py=filename_py,
            references=self.citations,
        )

        return card


distilabel.distiset.Distiset = CustomDistisetWithAdditionalTag
