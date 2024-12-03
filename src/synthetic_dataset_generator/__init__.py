import warnings
from typing import Optional

import distilabel
import distilabel.distiset
from distilabel.llms import InferenceEndpointsLLM
from distilabel.utils.card.dataset_card import (
    DistilabelDatasetCard,
    size_categories_parser,
)
from huggingface_hub import DatasetCardData, HfApi
from pydantic import (
    ValidationError,
    model_validator,
)


class CustomInferenceEndpointsLLM(InferenceEndpointsLLM):
    @model_validator(mode="after")  # type: ignore
    def only_one_of_model_id_endpoint_name_or_base_url_provided(
        self,
    ) -> "InferenceEndpointsLLM":
        """Validates that only one of `model_id` or `endpoint_name` is provided; and if `base_url` is also
        provided, a warning will be shown informing the user that the provided `base_url` will be ignored in
        favour of the dynamically calculated one.."""

        if self.base_url and (self.model_id or self.endpoint_name):
            warnings.warn(  # type: ignore
                f"Since the `base_url={self.base_url}` is available and either one of `model_id`"
                " or `endpoint_name` is also provided, the `base_url` will either be ignored"
                " or overwritten with the one generated from either of those args, for serverless"
                " or dedicated inference endpoints, respectively."
            )

        if self.use_magpie_template and self.tokenizer_id is None:
            raise ValueError(
                "`use_magpie_template` cannot be `True` if `tokenizer_id` is `None`. Please,"
                " set a `tokenizer_id` and try again."
            )

        if (
            self.model_id
            and self.tokenizer_id is None
            and self.structured_output is not None
        ):
            self.tokenizer_id = self.model_id

        if self.base_url and not (self.model_id or self.endpoint_name):
            return self

        if self.model_id and not self.endpoint_name:
            return self

        if self.endpoint_name and not self.model_id:
            return self

        raise ValidationError(
            f"Only one of `model_id` or `endpoint_name` must be provided. If `base_url` is"
            f" provided too, it will be overwritten instead. Found `model_id`={self.model_id},"
            f" `endpoint_name`={self.endpoint_name}, and `base_url`={self.base_url}."
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
distilabel.llms.InferenceEndpointsLLM = CustomInferenceEndpointsLLM
