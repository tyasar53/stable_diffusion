# Usage Guide

This guide explains the different functionalities of the prototype and how to use them.

## Making Inference

To generate images (i.e. make inference), the first step is to load a model.

### Loading a Model

To be able to make inference, select a model from the *Model Dropdown* and press the **Load Selected Model** button. After successfully loading, the **Inference** tab should load the UI elements to generate images.

### See the Concepts the Selected Model is Able to Generate

Before actually loading the model, you can inspect the concepts with their names and pictures that the selected model has learned. To do this, pick a model from the **Model Dropdown** and press the **Show Concepts of Selected Model** button.

If you want to hide the concepts, clicking on another model from the **Model Dropdown** list is sufficient.

### Prompt Generator

If you want to generate a prompt to generate images, you can use the *Prompt Generator* provided at the top of the **Inference** tab. If you want to generate images based on your own text, you can skip this part and proceed to *Generating Images* below.

After the model has successfully loaded, you should see that the **Inference** tab has been populated, and a **Prompt Generator** is provided. It is there to help you generate prompts by including properties of the Concepts and also introducing properties of general fashion elements.

### Selecting a Base Piece

Here you will select the fashion piece to which you can apply styles. For this, you have two choices. You can either define a general piece (e.g., t-shirt, dress, boots, etc.) by providing it in the textbox *Base Piece* or you can select a concept from the Dropdown *Base Concept*. If the textbox is filled with an input, the **Base Concept** dropdown will be ignored for the prompt generation.

### Including Properties of the Concepts

For each concept that the model has learned, there will be a *Concept* dropdown, and below that a group of checkboxes *Elements of the Concept* where you can select which elements of the concept should be included in the prompt (e.g., shape, pattern, color, etc.).

### Including Properties of General Elements

For elements of a fashion piece (e.g., shape, pattern, color), you will have a respective group of checkboxes *Properties of XXX* that allows you to include the property of those elements (e.g., shape has the properties oversize, skinny, etc.).

After you have made your selections, you can click the *Generate Prompt* button. This will automatically generate a prompt and put it into the field *Prompt*. The prompt will include all your selections. You are free to modify the prompt after generating it. By pressing the *Generate Image from Prompt* button, the model will generate images based on your prompt.

## Generating Images

Either by using the *Prompt Generator* or by crafting your own text, you can instruct the model to generate images. In the following, we will explain the options provided for the generation process.

### Prompt

This textbox is the main ingredient for the text-guided image generation. Whatever goes in here is the main condition on which images are generated.

### Negative Prompt

This textbox is passed along with the prompt. It will exclude everything in the image generation that has been provided in this field.

### Number of Samples

The prompt or prompts to guide the image generation. The more samples you ask the model to produce, the longer the inference time will be. Tip: Keep this number low until you find promising results. It might take some modification of the prompt to catch the desired output. By increasing the number, you will get more samples of what you are actually satisfied with.

### Guidance Scale

Higher guidance scale encourages generating images closely linked to the text prompt, usually at the expense of lower image quality.

Tip: If you feel that the model didn't really understand your prompt, increase this number.

### Height and Width

The height and width in pixels of the generated image. It has to be a multiple of 8. It does not have to be quadratic. The higher the image resolution, the longer the inference time.

### Steps

The number of denoising steps. More denoising steps usually lead to a higher-quality image at the expense of slower inference.

Tipp: To achieve proper results, the step size should not be lower than 50. Increasing to 200 to 400 steps usually leads to a perceivable better image quality. After step sizes 500 to 650, the increase in quality is not as conceivable anymore. These are rough estimates and depend on the richness of detail of the desired output.

## Regenerate from Output

If you are not satisfied with the output, you can either generate again or you can take an image you like and feed it into the Image-to-Image pipeline. For this, you drag and drop the desired image from the *Generating Images* section of the *Gallery* component into the *Image* field of this section. Then you can use the options, which are similar to the ones from the *Generating Images* section. The functionality of the provided options here is the same as in the *Generating Images* section except for the additional *Strength* option.

**ATTENTION: Drag and drop only works in the Safari web browser. If you use Chrome or Firefox, you have to right-click the image you want to regenerate on, save it to your computer, and re-upload it to the image component of the Regenerate from Output section.**

### Strength

The strength conceptually indicates how much to transform the reference image. It must be between 0 and 1. The image will be used as a starting point, adding more noise to it the larger the strength. The number of denoising steps depends on the amount of noise initially added. When strength is 1, added noise will be maximum, and the denoising process will run for the full number of iterations specified in **Steps**. A value of 1, therefore, essentially ignores the image.

## Train New Concepts

This tab provides the option to train your own concepts. You have two options to pick a base model on which your concepts will be trained.

You can either tick the checkbox at the top or leave the *Model Dropdown* selection at the very top empty. This will then use the `runwayml/stable-diffusion-v1-5` model provided on [huggingface.co](http://huggingface.co/) as your base model.

Or you can pick a model from the *Model Dropdown* and do not tick the checkbox at the top of the *Train New Concepts* tab. This will then retrain an already trained model. Your concept will then be added to the existing ones.

**ATTENTION: If you pick an already trained model as a base model, make sure to pick different names for your concepts as the ones already known by the model.**

### Providing Your Concept(s)

The layout for uploading your concepts consists of three building blocks that we will explain now. In total, you have **five** of them. Meaning that you can teach your model five concepts at a time.

### File

Here you can upload the images of your concept. A concept is a single entity. The images should, therefore, ideally represent the same entity but from different views and perspectives. In general, a concept can be anything. From a fashion piece up to an inspirational painting or pattern.

**ATTENTION: Do only upload images here. It is technically possible to upload any filetype. However, if non-image files (everything except png or jpeg) are uploaded, when training starts training, this will cause an error.**

### Name of Your Concept

This is the name that you will give your concept and will be used in the prompts to refer to your concepts during inference. Ideally, it should be a word that the model probably has a low prior knowledge of. At the same time, **it should not be a bunch of random characters**. If the name is too cryptic, the model will try to print the characters into the image.

### Class

The class name should be a single-worded noun that best describes the category your concept belongs to (e.g., t-shirt, sneaker, painting, etc.). Providing an accurate class name is crucial to capture the essence of your concept and use it in the right context during image generation.

### Model Name

This will be the name of your model. After training successfully, you are able to see this model in the *Model Dropdown* selection at the very top of your application.

**ATTENTION: To see it in the *Model Dropdown*, you have to click the *Refresh List* button at the very top of the application.**

### Number of Training Steps

The number of steps is the number of optimization steps during training.