
import gradio as gr
import os
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler, StableDiffusionImg2ImgPipeline
from PIL import Image
import shutil
from zipfile import ZipFile
from helper_functions import construct_concept_objects, construct_json, upload_files, empty_data_dir, adapt_training_config, get_available_models, log_training, get_concepts
from train_dreambooth import main

MAX_CONCEPTS_TO_SHOW=30


#TO DO 
# uploading multiple images, training on the lattest model

pipe_text2img = None
pipe_img2img = None
g_cuda = torch.Generator(device='cuda')

def inference(prompt, negative_prompt, num_samples, height=512, width=512, num_inference_steps=50, guidance_scale=7.5):

    #gets the information from the UI to generate images from text

    if (height % 8) != 0 or (width % 8): #check if output image dimensions are mutliples of 8
        raise gr.Error("Both height and width must be multiples of 8")
        return

    with torch.autocast("cuda"), torch.inference_mode():
        return pipe_text2img(
                prompt, height=int(height), width=int(width),
                negative_prompt=negative_prompt,
                num_images_per_prompt=int(num_samples),
                num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                generator=g_cuda
            ).images

def img2img(prompt, negative_prompt, image, num_samples, guidance_scale=7.5, num_inference_steps=50, strength=0.8):

    #gets the infromation from the UI to genrate images from images and from text

  with torch.autocast("cuda"), torch.inference_mode():
        return pipe_img2img(
                prompt, image,
                negative_prompt=negative_prompt,
                num_images_per_prompt=int(num_samples),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=guidance_scale,
                strength=strength
            ).images

def init_model(weights_path: str):

    #after a user as selected a model the model is initiliazied and can then be used to make inference by the user

    if weights_path == "":
        raise gr.Error("No model selected. Please choose from dropdown menu")
        return
    if isinstance(weights_path, list):
        weights_path = weights_path[0]
    weights_path = weights_path   # If you want to use previously trained model saved in gdrive, replace this with the full path of model in gdrive
    global pipe_img2img
    global pipe_text2img

    pipe_text2img         = StableDiffusionPipeline.from_pretrained(weights_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(weights_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
    
    pipe_text2img.scheduler = DDIMScheduler.from_config(pipe_text2img.scheduler.config)
    pipe_img2img.scheduler = DDIMScheduler.from_config(pipe_img2img.scheduler.config)

    #pipe.enable_xformers_memory_efficient_attention()
    #pipe_img2img.enable_xformers_memory_efficient_attention()


    concepts = get_concepts(weights_path)
 
    return { #you can return object. Keys named same as in outputs parameter of calling event of this function
        model_load_progress: gr.update(value=f"Successfuly loaded the model: {weights_path}", visible=True),
        text2img_inference: gr.update(visible=True),
        img2img_inference:  gr.update(visible=True),
        prompt_generator: gr.update(visible=True)
        }

def train(concept_images_1, concept_images_2, concept_images_3,
            concept_name_1, concept_name_2, concept_name_3,
            concept_class_name_1, concept_class_name_2, concept_class_name_3,
            model_name, max_train_steps, weights, use_base_model_checkbox):
    """
    gets the concept images, name of the concept and the class names of the concepts to tr
    """
    files           = [concept_images_1,        concept_images_2,       concept_images_3]
    concept_names   = [concept_name_1,          concept_name_2,         concept_name_3]
    class_names     = [concept_class_name_1,    concept_class_name_2,   concept_class_name_3]

    concepts = construct_concept_objects(files, concept_names, class_names)
    if construct_json(concepts) != None:
        raise gr.Error("Please Provide Concept data")
        return

    upload_files(concepts)
    adapt_training_config(model_name, max_train_steps, weights, use_base_model_checkbox)
    main() #start training of the model
    log_training(model_name, weights, use_base_model_checkbox) #to be able to load the input images of each concepts later

    return gr.Textbox.update(value=f"Successfully Trained model {model_name}")
 
def variable_outputs(model_path):
        if model_path == "":
            raise gr.Error("No model selected. Please choose from dropdown menu")
            return
        concepts = get_concepts(model_path)
        k = len(concepts)
        update_textboxes = []
        update_galleries = []
        for i in range(k):
            update_textboxes.append(gr.Textbox.update(visible=True, value=concepts[i]["instance_prompt"]))
            update_galleries.append(gr.Gallery.update(visible=True, value=concepts[i]["instance_images"]))

        return [gr.Markdown.update(visible=True)]+update_textboxes + [gr.Textbox.update(visible=False)]*(MAX_CONCEPTS_TO_SHOW - k) + update_galleries + [gr.Gallery.update(visible=False)] * (MAX_CONCEPTS_TO_SHOW - k) + [gr.Textbox.update(value="Showing Concepts for model below")]
        #return {
         #   markdown_concept: gr.Markdown.update(visible=True)
          #  textboxes_concept: update_textboxes + [gr.Textbox.update(visible=False)]*(MAX_CONCEPTS_TO_SHOW - k)
          #  galleries_concept: update_galleries + [gr.Gallery.update(visible=False)] * (MAX_CONCEPTS_TO_SHOW - k)
          #  }

def disable_concepts_view():
        update_textboxes = []
        update_galleries = []
        for i in range(MAX_CONCEPTS_TO_SHOW):
            update_textboxes.append(gr.Textbox.update(visible=False))
            update_galleries.append(gr.Gallery.update(visible=False))

        return [gr.Markdown.update(visible=False)] + update_textboxes  + update_galleries + [gr.Textbox.update(value="", visible=False)]
  
def update_dropdown_options(dropdown):
    choices = get_available_models()
    return gr.Dropdown.update(choices=choices)
    
def update_concept_dropdown(weights_path):
    concepts = get_concepts(weights_path)
    options_dropdown = []
    k = len(concepts)
    for i in range(len(concepts)):
        options_dropdown.append(concepts[i]["instance_prompt"].replace('photo of', ''))
    return [gr.Dropdown.update(choices=options_dropdown, visible=True, value="")]* (k+1) + [gr.Dropdown.update(visible=False, value="")]*(MAX_CONCEPTS_TO_SHOW - k) + [gr.CheckboxGroup.update(visible=True, interactive=True, value=[])] * k + [gr.CheckboxGroup.update(visible=False, value=[])]*(MAX_CONCEPTS_TO_SHOW - k)

def generate_prompt(
    bp_text, bp_concept,
    d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30,
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30,
    txt_shape, txt_pattern, txt_texture, txt_color, txt_space,
    cb_shape, cb_pattern, cb_texture, cb_color, cb_space,
    custom_shape, custom_pattern, custom_texture, custom_color, custom_space):
    #there are as many dropdowns as there are checkboxes. The first half are all dropdowns, the second half are all checkboxes
    fashion_elements      = [txt_shape, txt_pattern, txt_texture, txt_color, txt_space]
    fashion_element_props = [cb_shape , cb_pattern , cb_texture , cb_color , cb_space ]
    fashion_element_custom = [custom_shape, custom_pattern, custom_texture, custom_color, custom_space]
    dropdowns =  [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30]
    checkboxes = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30]

    prompt = f"photo of {bp_text}" if len(bp_text) > 0 else f"photo of {bp_concept}"

    for i in range(len(dropdowns)):
        if dropdowns[i] != "" and len(checkboxes[i]) > 0: #so if there is an option selected
            prompt += " with "
            if len(checkboxes[i]) == 1: #if it is only a single option that is selected
                prompt += f"{checkboxes[i][0]} of {dropdowns[i]} and" 
            else: #if there are more then one option selected
                for j in range(len(checkboxes[i])):
                    prompt += f"{str(checkboxes[i][j])}, " if len(checkboxes[i]) != (j+1) else f" and {checkboxes[i][j]}"

                prompt += f" of {dropdowns[i]} and"

    for i in range(len(fashion_elements)):
        if len(fashion_element_custom[i]) > 0:
            prompt += f" with {fashion_element_custom[i]} {fashion_elements[i]} and"
            continue

        if len(fashion_element_props[i]) > 0:
            prompt += " with "
            if len(fashion_element_props[i]) == 1:
                prompt += f"{fashion_element_props[i][0]} {fashion_elements[i]} and" 
            else:
                for j in range(len(fashion_element_props[i])):
                    print(str(fashion_element_props[i][j]))
                    print(str(fashion_element_props[i]))

                    prompt += f"{ str(fashion_element_props[i][j]) }, " if len(fashion_element_props[i]) != (j+1) else f" and {str(fashion_element_props[i][j])}"
                prompt += f" {str(fashion_elements[i])} and"
                    

    prompt = prompt[:-3] if prompt[-3:] == "and" else prompt

    return prompt #except the last 'and'


dropdown_options = get_available_models()

with gr.Blocks() as demo:
    gr.Markdown('# Stable Diffusion / Dreambooth - Prototype')

    #Model Selection and Controll
    with gr.Row():
        with gr.Column():
            gr.Markdown('Select the model that you want to use to make inference from')
            model_dropdown = gr.Dropdown(choices=dropdown_options, interactive=True, multiselect=False)
        with gr.Column():
            #Refresh List
            gr.Markdown('Press this button after training your own concept. It refreshes the list of availabel models')
            btn_dropdown_refresh = gr.Button(value="Refresh List")

            #Show Concepts Selected
            gr.Markdown("Press this button to show the trained concepts on the model that you have selected from the dropdown")
            btn_show_concepts    = gr.Button(value="Show Concepts Selected Model")
            txt_loading_show_concepts  = gr.Textbox(label="loading status", value="", interactive=False, visible=False)
            btn_show_concepts.click(fn=lambda: gr.update(visible=True), outputs=[txt_loading_show_concepts])
       
            #Load Selected Model
            gr.Markdown("Press this button AFTER you have selecte a model from the dropdowon")
            model_load_button    = gr.Button(value="Load Selected Model")
            model_load_progress  = gr.Textbox(label="loading status", value="", interactive=False, visible=False)
            model_load_button.click(fn=lambda: gr.update(visible=True), outputs=[model_load_progress])


        btn_dropdown_refresh.click(update_dropdown_options, outputs=[model_dropdown])
    
    #Concepts of a Model 
    with gr.Row():
        markdown_concept = gr.Markdown('ATTENTION: This selection will be shown if you select a model. If you want to make inference with this model you always have to clicke the buttion "Load Selected Model"', visible=False)
        textboxes_concept = []
        galleries_concept = []
        for i in range(MAX_CONCEPTS_TO_SHOW):
            t = gr.Textbox(f"Textbox {i}", visible=False, interactive=False)
            g = gr.Gallery(visible=False)
            textboxes_concept.append(t)
            galleries_concept.append(g)

        model_dropdown.change(disable_concepts_view, outputs=[markdown_concept]+ textboxes_concept+galleries_concept + [txt_loading_show_concepts])
        btn_show_concepts.click(variable_outputs, inputs=[model_dropdown], outputs=[markdown_concept]+ textboxes_concept+galleries_concept + [txt_loading_show_concepts])

        
    #Inference Tab
    with gr.Tab("Inference"):
        with gr.Row(visible=False) as prompt_generator:
            with gr.Column():
                gr.Markdown("## Prompt Generator")
                gr.Markdown("with the provide elemenst below you can generate prompts that combine different aspects of two concepts")
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            gr.Markdown("If text box 'Base Piece' is filled the selected concept on the right is ignored")
                        with gr.Row():
                            base_piece_prompt_generator_text = gr.Textbox(label="Base Piece", info="This is the piece of cloths that the transfer of styles is applied to. For example, writing 'white thshirt' will generate a prompt 'photo of a white tshirt with shape of concept1 and color of concept2 ...")
                            base_piece_prompt_generator_concept = gr.Dropdown(label=" concept", interactive=True, multiselect=False,visible=True)
                        with gr.Row():
                            gr.Markdown("")
                        with gr.Row():
                            dropdowns_prompt_generator = []
                            checkboxes_prompt_generator= []
                            for i in range(30): #has to be 30, because of number of attributes generate propmt function takes
                                with gr.Column():
                                    dropdowns_prompt_generator.append(gr.Dropdown(label=" concept", interactive=True, multiselect=False,visible=True))
                                    checkboxes_prompt_generator.append(gr.CheckboxGroup(["shape", "pattern", "color", "length"], label="Elements of the Concept", info="Select the elements of the concept above that should be included e.g. clicking on pattern shape will add 'with pattern of XXX XXX'", visible=True))
                        with gr.Row():
                            fashion_elements = [
                                {
                                    "name": "shape",
                                    "properties": [
                                        'oversize', 'skinny', 'baloon'
                                    ]
                                },
                                {
                                    "name": "pattern",
                                    "properties": [
                                        'stripes', 'animal', 'print', 'floral'
                                    ]
                                },
                                {
                                    "name": "texture",
                                    "properties": [
                                        'leather', 'velvet', 'fleece'
                                    ]
                                },                                {
                                    "name": "color",
                                    "properties": [
                                        'red', 'green', 'blue', 'white', 'yellow', 'pink', 'purple'
                                    ]
                                },
                                {
                                    "name": "space",
                                    "properties": [
                                        'positive background', 'negative background'
                                    ]
                                }
                                ]
                            textbox_fashion_elements = []
                            checkbox_fashion_elements= []
                            textbox_fashion_element_custom=[]
                            for fashion_element in fashion_elements:
                                with gr.Column():
                                    textbox_fashion_elements.append(gr.Textbox(label="Element", value=fashion_element['name'], interactive=False))
                                    checkbox_fashion_elements.append(gr.CheckboxGroup(fashion_element['properties'], label=f"Properties of {fashion_element['name']}", info=f"Selecting this will include to the prompt e.g. 'and with ... {fashion_element['name']}'"))
                                    textbox_fashion_element_custom.append(gr.Textbox(label="custom property", interactive=True, info="if this field is filled the above checkboxes are ignored"))
                        with gr.Row():
                            btn_combine = gr.Button(value="Generate Prompt")
                            #txt_generated_prompt = gr.Textbox(label="Generated Prompt")


        with gr.Row(visible=False) as text2img_inference:
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="photo of lur dress", info=" The prompt or prompts to guide the image generation")
                negative_prompt = gr.Textbox(label="Negative Prompt", value="", info="The prompt or prompts not to guide the image generation")
                run = gr.Button(value="Generate")
                with gr.Row():
                    num_samples = gr.Number(label="Number of Samples", value=4, info="Provide the number of samples that the model generates")
                    guidance_scale = gr.Number(label="Guidance Scale", value=7.5, info="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality")
                with gr.Row():
                    height = gr.Number(label="Height", value=512, info="The height in pixels of the generated image. Has to be multiple of 8")
                    width = gr.Number(label="Width", value=512, info="The width in pixels of the generated image. Has to be mutliple of 8")
                num_inference_steps = gr.Slider(label="Steps", value=50, maximum=1000, info="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference")
            with gr.Column():
                gallery = gr.Gallery()
        with gr.Row(visible=False) as img2img_inference:
            with gr.Column():
                gr.Markdown("## Regenerate from Output")
                prompt_img2img          = gr.Textbox(label="Img2Img Prompt", value="change the color to blue", interactive=True, info="The prompt or prompts to guide the image generation")
                negative_prompt_img2img = gr.Textbox(label="Img2Img Negative Prompt", value="", interactive=True, info="The prompt or prompts not to guide the image generation" )
                regenerate = gr.Button("Regenerate")
                with gr.Row():
                    num_samples_img2img    = gr.Number(label="Number of Samples", value=4, info="Provide the number of samples that the model generates")
                    guidance_scale_img2img = gr.Number(label="Guidance Scale", value=7.5, info="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality")  
                with gr.Row():
                    num_inference_steps_img2img = gr.Slider(label="Steps", value=50, maximum=1000, info="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference")
                    strength_img2img = gr.Slider(label="Strength", value=0.8, minimum=0 ,maximum=1, info=" Conceptually, indicates how much to transform the reference image. Must be between 0 and 1. image will be used as a starting point, adding more noise to it the larger the strength. The number of denoising steps depends on the amount of noise initially added. When strength is 1, added noise will be maximum and the denoising process will run for the full number of iterations specified in **Steps**. A value of 1, therefore, essentially ignores image.")

            with gr.Column():
                gr.Markdown('Drag & Drop the output from the inference here. This serves as input for the regeneration based on the prompt provided on the left')
                input_img2img = gr.Image(type="pil", download=True)
                input_img2img.download = True
                gallery_img2img = gr.Gallery(type="pil")
    
    #Training Concepts Tab
    with gr.Tab('Train New Concept'):
        with gr.Column() as training_concepts:
            with gr.Row():
                gr.Markdown(
                '''
                **File:**
                Plese upload the image files for the concept you want to teach the model. DO ONLY UPLOAD IMAGE FILE (png, jpeg)


                **Name of your concept:** 
                is the name that you will give your concept it should be a word that the model probably has a low prior knowledge. At the same time it should not be a bunch of random characters. If the name is to cryptic, the model will try to print the characters into the image
                
                
                **Class:** 
                The class name should be a single worded noun that best describes the category you concept belongs to.
                ''')

            use_base_model_checkbox = gr.Checkbox(label='Use new model for training (if not selected the model selecte from the dropdown above will be selected. If that is left empty then a new model will be used for training)', interactive=True)
            with gr.Row():
                concept_images_1 = gr.File(file_count='multiple', interactive=True) 
                concept_name_1 = gr.Textbox(label="Name of your concept", value="coco", interactive=True)
                concept_class_name_1   = gr.Textbox(label="Class", value="jacket", interactive=True)
            with gr.Row():
                concept_images_2 = gr.File(file_count='multiple', interactive=True) 
                concept_name_2 = gr.Textbox(label="Name of your concept", value="coco", interactive=True)
                concept_class_name_2   = gr.Textbox(label="Class", value="jacket", interactive=True)
            with gr.Row():
                concept_images_3 = gr.File(file_count='multiple', interactive=True) 
                concept_name_3 = gr.Textbox(label="Name of your concept", value="coco", interactive=True)
                concept_class_name_3   = gr.Textbox(label="Class", value="jacket", interactive=True)
            
            model_name   = gr.Textbox(label="Model Name", value="my-model", interactive=True)
            gr.Markdown("Number of optimization steps. A value between 200 and 400 is sufficient in most cases")
            max_train_steps = gr.Number(label="Number of training steps", value=400, interactive=True)

            train_button = gr.Button(value="Train New Concepts", )
            concept_textbox = gr.Textbox(visible=True)
            remove_data_button = gr.Button(value="Remove Data")
            result_remove_operation = gr.Textbox(label="Result of remove operation", interactive=False)
            train_button.click(fn=lambda x: gr.Textbox.update(visible=True))

    #Event Listeners
    train_button.click(train, 
        inputs=[
            concept_images_1, concept_images_2, concept_images_3,
            concept_name_1, concept_name_2, concept_name_3,
            concept_class_name_1, concept_class_name_2, concept_class_name_3,
            model_name, max_train_steps, model_dropdown, use_base_model_checkbox
        ], 
        outputs=[concept_textbox])
    run.click(inference, inputs=[prompt, negative_prompt, num_samples, height, width, num_inference_steps, guidance_scale], outputs=gallery)
    regenerate.click(img2img, inputs=[prompt_img2img, negative_prompt_img2img, input_img2img, num_samples_img2img, guidance_scale_img2img, num_inference_steps_img2img, strength_img2img], outputs=gallery_img2img)
    remove_data_button.click(empty_data_dir, outputs=[result_remove_operation])
    model_load_button.click(init_model, inputs=[model_dropdown], outputs=[model_load_progress, text2img_inference, img2img_inference, prompt_generator])
    btn_dropdown_refresh.click(update_dropdown_options, inputs=[model_dropdown])




    #event listeners for  Prompt Generator 
    model_dropdown.change(update_concept_dropdown, inputs=[model_dropdown], outputs=[base_piece_prompt_generator_concept] + dropdowns_prompt_generator + checkboxes_prompt_generator)
    btn_combine.click(generate_prompt, inputs=[base_piece_prompt_generator_text, base_piece_prompt_generator_concept] + dropdowns_prompt_generator+checkboxes_prompt_generator + textbox_fashion_elements + checkbox_fashion_elements + textbox_fashion_element_custom , outputs=[prompt] )

demo.launch(debug=True, share=True)