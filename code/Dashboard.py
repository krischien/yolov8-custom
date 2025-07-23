import gradio as gr

def greet(image, slider1, slider2, slider3):
    return f"Slider values: {slider1}, {slider2}, {slider3}"

demo = gr.Interface(
    fn=greet,
    inputs=[
        gr.Image(type="numpy", label="Input Source"),
        gr.Slider(minimum=0, maximum=100, label="Image Size"),
        gr.Slider(minimum=0, maximum=100, label="Confidence Threshold"),
        gr.Slider(minimum=0, maximum=100, label="IOU Threshold"),
    ],
    outputs=[]  # Added label for output
)

demo.launch()

