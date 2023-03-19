
import gradio as gr
from PIL import ImageOps

def inference(prompt, negative_prompt, num_samples, height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
    pass

def img2img(image):
    image = image.split()[-1]
    image = ImageOps.invert(image)
    image = image.convert('L')
    image = image.point(lambda x: 255 if x > 0 else 0, mode='1')

    return image


with gr.Blocks() as demo:
   with gr.Row():
      pad    = gr.Sketchpad(label='Sketch', shape=(400, 400))
      image  = gr.Image('dress2.jpg')
      sketch = gr.Image(label="Image for img2img", elem_id="img2img_sketch", show_label=False, source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="L").style(height=1000)
      button = gr.Button(value='Seperate')
      result = gr.Image()

      button.click(img2img, inputs=sketch, outputs=result)

'''
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="photo of zwx dog in a bucket")
            negative_prompt = gr.Textbox(label="Negative Prompt", value="")
            run = gr.Button(value="Generate")
            with gr.Row():
                num_samples = gr.Number(label="Number of Samples", value=4)
                guidance_scale = gr.Number(label="Guidance Scale", value=7.5)
            with gr.Row():
                height = gr.Number(label="Height", value=512)
                width = gr.Number(label="Width", value=512)
            num_inference_steps = gr.Slider(label="Steps", value=24)
        with gr.Column():
            with gr.Row():
                gallery = gr.Gallery()
                prompt_img2img          = gr.Textbox(label="Img2Img Prompt", value="change the color to blue")
                negative_prompt_img2img = gr.Textbox(label="Img2Img Negative Prompt", value="")
            with gr.Row():
                num_samples_img2img    = gr.Number(label="Number of Samples", value=4)
                guidance_scale_img2img = gr.Number(label="Guidance Scale", value=7.5)  
            with gr.Row():
                impaint = gr.Button("Apply Impainting")
            with gr.Row():
                input_img2img = gr.Image(type="pil")
            with gr.Row():
                gallery_img2img = gr.Gallery()



    

    run.click(inference, inputs=[prompt, negative_prompt, num_samples, height, width, num_inference_steps, guidance_scale], outputs=gallery)
    impaint.click(img2img, inputs=[prompt_img2img, negative_prompt_img2img, input_img2img, num_samples_img2img], outputs=gallery_img2img)
'''

demo.launch(debug=True)