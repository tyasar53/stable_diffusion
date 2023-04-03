
# Computational Creativity: Generative Fashion Design with AI-Driven System — Prototype

This is the documentation for the prototype of the Team Project 2022/2023 with the chair of Prof. Dr. A. Heinzel at the University of Mannheim. To know how to use this model, please refer to the [Usage Guide](./docs/usage-guide.md). The code for training new models was inspired by this repository of Shivam Shrirao from [GitHub](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth).

## Installation

### Install requirements

To create a virtual environment and install all dependencies from the [requirements.txt](./requirements.txt) file, follow these steps:

1. Open your terminal or command prompt.
2. Navigate to the root of this project.
3. Type `python3 -m venv venv` and press Enter. This will create a new virtual environment named 'venv'.
4. Activate the virtual environment by typing `source venv/bin/activate` on Linux 
5. Once the virtual environment is activated, type `pip install -r requirements.txt` and press Enter. This will install all dependencies listed in the requirements.txt file.

### Save Huggingface-Token

To be able of dowloading models from huggingface.co you need to create a token first and the save it to the machine.

```bash
mkdir -p ~/.huggingface
echo -n "HUGGINGFACE_TOKEN" > ~/.huggingface/token
```

the programm will then automatically detect it and use it to retrieve models from huggingface.co

Before starting the application you have to create a folder called weights.

```bash
mkdir weights
```

## Starting the Application

After completing the above steps you can start the application

```bash
python app.py
```
If you want to develop the prototype further, by running 

```bash
gradio app.py
```

it will start the applicaiton in reload mode. Whenever something is changed, the applicaiton reloads automatically to show the results of you changes immediately. 
adding the weights

## Sharing the application
in the [app.py](app.py) file at the very top you can set the GRADIO_SHARE variable to True. Whenever starting the application it generates a random link that makes the application accessible from the internet. However if you reload the application the link changes.
If you want a stable permanent link you can refer to [ngrok](https://ngrok.com/). They provide easy to setup tunnels to you application with permanent links.

## Use pre-trained models
to be able to use the pretrained models that we used for testing you basically have to download the first model ([model-a](https://drive.google.com/drive/folders/1JE81HP16Hp0RbmbtRQzmCMHMnmATZ9_0?usp=share_link)) with 20+ concepts and the second model ([model-b](https://drive.google.com/drive/folders/1hLI7l3Ao0OKls0r5OA8MLuqt9wM9RmO5?usp=share_link)) with 6 concepts.
After downloading you have to put booth folders "model-a" and "model-b" in the [weights](./weights) folder of this project. Make sure that they are actually called "model-a" and "model-b". Start the application and you should be able to select the two models from the dropdown. After loading them in the application you are ready to generate images from the concepsts.

## Info
Make sure to have sufficiently large memory and gpu access to run this model. 20 GB of VRAM are required at least.