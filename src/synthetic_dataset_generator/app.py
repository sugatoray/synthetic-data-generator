from synthetic_dataset_generator._tabbedinterface import TabbedInterface
from synthetic_dataset_generator.apps.eval import app as eval_app
from synthetic_dataset_generator.apps.faq import app as faq_app
from synthetic_dataset_generator.apps.sft import app as sft_app
from synthetic_dataset_generator.apps.textcat import app as textcat_app

theme = "argilla/argilla-theme"

css = """
.main_ui_logged_out{opacity: 0.3; pointer-events: none}
"""

demo = TabbedInterface(
    [textcat_app, sft_app, eval_app, faq_app],
    ["Text Classification", "Supervised Fine-Tuning", "Evaluation", "FAQ"],
    css=css,
    title="Synthetic Data Generator",
    head="Synthetic Data Generator",
    theme=theme,
)


if __name__ == "__main__":
    demo.launch()
