import gradio as gr

from src.distilabel_dataset_generator.apps.faq import app as faq_app
from src.distilabel_dataset_generator.apps.sft import app as sft_app

theme = gr.themes.Monochrome(
    spacing_size="md",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

css = """
h1{font-size: 2em}
h3{margin-top: 0}
#component-1{text-align:center}
.main_ui_logged_out{opacity: 0.3; pointer-events: none}
.tabitem{border: 0px}
.group_padding{padding: .55em}
#space_model .wrap > label:last-child{opacity: 0.3; pointer-events:none}
"""

demo = gr.TabbedInterface(
    [sft_app, faq_app],
    ["Supervised Fine-Tuning", "FAQ"],
    css=css,
    title="⚗️ distilabel Dataset Generator",
    head="⚗️ distilabel Dataset Generator",
    theme=theme,
)


if __name__ == "__main__":
    demo.launch()
