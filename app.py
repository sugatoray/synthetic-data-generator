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
    <style>
        @media (max-width: 600px) {
            .logo-container { position: static !important; text-align: center; margin-bottom: 20px; }
            .title-container { padding-left: 0 !important; }
        }
    </style>
    <div style="position: relative; margin-bottom: 1rem;">
        <div class="logo-container" style="position: absolute; top: 0; left: 0;">
            <img src="https://distilabel.argilla.io/latest/assets/distilabel-black.svg" alt="Distilabel Logo" style="width: 150px; height: auto;">
        </div>
        <div class="title-container" style="text-align: center; padding-top: 40px; padding-left: 160px;">
            <h1 style="margin: 0; font-size: 2em;">🧶 DataCraft</h1>
            <p style="margin: 10px 0 0 0; color: #666; font-size: 1.1em;">Build datasets using natural language</p>
        </div>
    </div>
    """,
    theme=theme,
)

if __name__ == "__main__":
    demo.launch()
