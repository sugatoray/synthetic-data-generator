from synthetic_dataset_generator._tabbedinterface import TabbedInterface
from synthetic_dataset_generator.apps.eval import app as eval_app
from synthetic_dataset_generator.apps.readme import app as readme_app
from synthetic_dataset_generator.apps.sft import app as sft_app
from synthetic_dataset_generator.apps.textcat import app as textcat_app

theme = "argilla/argilla-theme"

css = """
.main_ui_logged_out{opacity: 0.3; pointer-events: none}
"""

image = """<br><img src="https://raw.githubusercontent.com/argilla-io/synthetic-data-generator/main/assets/logo.svg" alt="Synthetic Data Generator Logo" style="display: block; margin-left: auto; margin-right: auto; width: 50%;"/>"""

demo = TabbedInterface(
    [textcat_app, sft_app, eval_app, readme_app],
    ["Text Classification", "Supervised Fine-Tuning", "Evaluation", "README"],
    css=css,
    title=image,
    head="Synthetic Data Generator",
    theme=theme,
)


if __name__ == "__main__":
    demo.launch()
