import gradio as gr

from src.distilabel_dataset_generator.faq import app as faq_app
from src.distilabel_dataset_generator.sft import app as sft_app

demo = gr.TabbedInterface(
    [sft_app, faq_app],
    ["Supervised Fine-Tuning", "FAQ"],
    title="⚗️ Distilabel Dataset Generator",
    head="⚗️ Distilabel Dataset Generator",
    theme="ParityError/Interstellar",
)

if __name__ == "__main__":
    demo.launch()
