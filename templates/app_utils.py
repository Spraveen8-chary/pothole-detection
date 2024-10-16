import gradio as gr

def detect_pothole(image):
    result = "Pothole detected!" if image is not None else "No pothole detected."
    return f"<h1 style='text-align: center; color: red;'>{result}</h1>"

with gr.Blocks() as interface:
    gr.Markdown("<h1 style='text-align: center; color: red;'>Pothole Detection System</h1>")
    gr.Markdown("<p style='text-align: center;'>Upload an image to detect potholes using our AI-powered system</p>")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", type="filepath")
        with gr.Column():
            output_text = gr.HTML()  

    detect_button = gr.Button("Detect Pothole")

    detect_button.click(detect_pothole, inputs=[input_image], outputs=[output_text])

interface.launch()
