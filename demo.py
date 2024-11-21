import gradio as gr

from src.distilabel_dataset_generator._tabbedinterface import TabbedInterface
from src.distilabel_dataset_generator.apps.eval import app as eval_app
from src.distilabel_dataset_generator.apps.faq import app as faq_app
from src.distilabel_dataset_generator.apps.sft import app as sft_app
from src.distilabel_dataset_generator.apps.textcat import app as textcat_app

theme = gr.themes.Monochrome(
    spacing_size="md",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

css = """
.main_ui_logged_out{opacity: 0.3; pointer-events: none}
.tabitem{border: 0px}
.group_padding{padding: .55em}
#space_model .wrap > label:last-child{opacity: 0.3; pointer-events:none}
#system_prompt_examples {
    color: black;
}
@media (prefers-color-scheme: dark) {
    #system_prompt_examples {
        color: white;
        background-color: black;
    }
}
button[role="tab"].selected,
button[role="tab"][aria-selected="true"],
button[role="tab"][data-tab-id][aria-selected="true"] {
    background-color: #000000;
    color: white;
    border: none;
    font-size: 16px;
    font-weight: bold;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    transition: background-color 0.3s ease, color 0.3s ease;
}
.gallery {
    color: black !important;
}
.flex-shrink-0.truncate.px-1 {
    color: black !important;
}
"""

demo = TabbedInterface(
    [textcat_app, sft_app, eval_app, faq_app],
    ["Text Classification", "Supervised Fine-Tuning", "Evaluation", "FAQ"],
    css=css,
    title="""
    <h1>Synthetic Data Generator</h1>
    <h3>Build datasets using natural language</h3>
    """,
    head="Synthetic Data Generator",
    theme=theme,
)


if __name__ == "__main__":
    demo.launch()
