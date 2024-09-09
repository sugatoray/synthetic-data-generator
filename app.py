import gradio as gr

from distilabel_dataset_generator.sft import demo

demo = gr.TabbedInterface(
    [demo],
    ["Supervised Fine-Tuning"],
    title="⚗️ Distilabel Dataset Generator",
    head="⚗️ Distilabel Dataset Generator",
)

demo.launch()
