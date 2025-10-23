import gradio as gr

with gr.Blocks() as demo:
    markdown = gr.Markdown(value="<h1>hi</h1>")

    @gr.on([], inputs=[], outputs=[])
    def fn_1():
        ...
        return 

demo.launch()