import gradio as gr

from src.distilabel_dataset_generator.apps.faq import app as faq_app
from src.distilabel_dataset_generator.apps.sft import app as sft_app

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
"""

demo = gr.TabbedInterface(
    [sft_app, faq_app],
    ["Supervised Fine-Tuning", "FAQ"],
    css=css,
    title="""
    <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; justify-content: center;">
            <img src="https://distilabel.argilla.io/latest/assets/distilabel-black.svg" alt="Distilabel Logo" style="width: 200px; height: auto;">
        </div>
        <p style="margin: 10px 0 0 0; font-style: italic; color: #666; font-size: 1.1em;">DataCraft: Build datasets using natural language</p>
    </div>
    """,
    theme=theme,
)

if __name__ == "__main__":
    demo.launch()
