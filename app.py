import gradio as gr

from distilabel_dataset_generator.sft import demo

demo = gr.TabbedInterface(
    [demo],
    ["Supervised Fine-Tuning"],
    title="⚗️ Distilabel Dataset Generator",
    head="⚗️ Distilabel Dataset Generator",
)

if __name__ == "__main__":
    demo.launch()
