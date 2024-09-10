import gradio as gr

from src.distilabel_dataset_generator.faq import app as faq_app
from src.distilabel_dataset_generator.sft import app as sft_app


theme = gr.themes.Monochrome(
    spacing_size="md",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
    
)

demo = gr.TabbedInterface(
    [sft_app, faq_app],
    ["Supervised Fine-Tuning", "FAQ"],
    title="⚗️ distilabel Dataset Generator",
    head="⚗️ distilabel Dataset Generator",
    theme=theme,
)


if __name__ == "__main__":
    demo.launch()
