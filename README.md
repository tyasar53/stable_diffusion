
# Fashion Design and AI: Generative Fashion Model based on Stable Diffusion

This is the documentation for the prototype. To know how to use this model, please refer to the [Usage Guide](./docs/usage-guide.md)

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

adding the weights