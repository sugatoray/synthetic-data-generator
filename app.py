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
@media (prefers-color-scheme: dark) {
    #system_prompt_examples {
        color: white;
        background-color: black;
    }
}
"""

demo = gr.TabbedInterface(
    [sft_app, faq_app],
    ["Supervised Fine-Tuning", "FAQ"],
    css=css,
    title="""
    <style>
        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            padding: 20px 0;
        }
        .logo-container {
            position: absolute;
            left: 0;
            top: 0;
        }
        .title-container {
            text-align: center;
        }
        @media (max-width: 600px) {
            .header-container {
                flex-direction: column;
            }
            .logo-container {
                position: static;
                margin-bottom: 20px;
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
    </style>
    <div class="header-container">
        <div class="logo-container">
            <a href="https://github.com/argilla-io/distilabel" target="_blank" rel="noopener noreferrer">
                <img src="https://distilabel.argilla.io/latest/assets/distilabel-black.svg" alt="Distilabel Logo" style="width: 150px; height: auto;">
            </a>
        </div>
        <div class="title-container">
            <h1 style="margin: 0; font-size: 2em;">🧬 Synthetic Data Generator</h1>
            <p style="margin: 10px 0 0 0; color: #666; font-size: 1.1em;">Build datasets using natural language</p>
        </div>
    </div>
    """,
    theme=theme,
)

if __name__ == "__main__":
    demo.launch()
