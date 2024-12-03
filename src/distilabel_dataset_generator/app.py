from distilabel_dataset_generator._tabbedinterface import TabbedInterface
from distilabel_dataset_generator.apps.eval import app as eval_app
from distilabel_dataset_generator.apps.faq import app as faq_app
from distilabel_dataset_generator.apps.sft import app as sft_app
from distilabel_dataset_generator.apps.textcat import app as textcat_app

theme = "argilla/argilla-theme"

css = """
button[role="tab"][aria-selected="true"] { border: 0; background: var(--neutral-800); color: white; border-top-right-radius: var(--radius-md); border-top-left-radius: var(--radius-md)}
button[role="tab"][aria-selected="true"]:hover {border-color: var(--button-primary-background-fill)}
button.hf-login {background: var(--neutral-800); color: white}
button.hf-login:hover {background: var(--neutral-700); color: white}
.tabitem { border: 0; padding-inline: 0}
.main_ui_logged_out{opacity: 0.3; pointer-events: none}
.group_padding{padding: .55em}
.gallery-item {background: var(--background-fill-secondary); text-align: left}
.gallery {white-space: wrap}
#space_model .wrap > label:last-child{opacity: 0.3; pointer-events:none}
#system_prompt_examples {
    color: var(--body-text-color) !important;
    background-color: var(--block-background-fill) !important;
}
.container {padding-inline: 0 !important}
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
