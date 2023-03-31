import os
import json
import shutil
import yaml
import json
from PIL import Image
import gradio as gr

def construct_concept_objects(files, concept_names, class_names):
    concepts = []
    for i in range(len(files)):
        if concept_names[i] != "" and class_names[i] != "" and files[i] != None:
            concepts.append({
                "idx": i,
                "concept_name": concept_names[i],
                "class_name":   class_names[i],
                "files": files[i]
                })
    return concepts

def construct_json(concepts):
    if len(concepts) < 1:
        return gr.Error('Please provide at least one concept to train (Images, Concept name, Class name)')
    

    concepts_list = []
    for c in concepts:

        concept_name = c["concept_name"]
        class_name = c["class_name"]

        concept = {
         "instance_prompt":      f"photo of {concept_name} {class_name}",
         "class_prompt":         f"photo of a {class_name}",
         "instance_data_dir":    f"{ os.path.join(os.getcwd(),f'data/{concept_name}')}",
         "class_data_dir":       f"{ os.path.join(os.getcwd(),f'data/{class_name}')}"
        }
        os.makedirs(concept["instance_data_dir"], exist_ok=True)
        concepts_list.append(concept)

        with open("concepts_list.json", "w") as f:
            json.dump(concepts_list, f, indent=4)
   
def upload_files(concepts):
    for concept in concepts:
        for idx, file in enumerate(concept["files"]):
            d = open(file.name, 'rb')
            d.seek(0)

            f = open(os.path.join(os.getcwd(),'data', concept["concept_name"], file.name.split('/')[-1]), "wb")
            f.write(d.read())
            
            d.close()
            f.close()

def empty_data_dir():
    data_dirs = next(os.walk('./data'))[1]
    for folder in data_dirs:
        shutil.rmtree(os.path.join(os.getcwd(), 'data', folder))
    return "All image folders have been removed"

def adapt_training_config(model_name, max_train_steps, weights, use_base_model_checkbox):
    with open(os.path.join(os.getcwd(), 'train.config.yml'), 'r') as stream:
        args=yaml.safe_load(stream)
    
    args['output_dir'] = os.path.join(os.getcwd(), 'weights', model_name)
    args['max_train_steps'] = int(max_train_steps)
    if weights != None and use_base_model_checkbox != True:
        args['pretrained_model_name_or_path'] = weights
    else:
        args['pretrained_model_name_or_path'] = 'runwayml/stable-diffusion-v1-5'

        

    with open(os.path.join(os.getcwd(), 'train.config.yml'), 'w') as config:
        config.write(yaml.dump(args))


def get_available_models():
    models = []
    for i in next(os.walk('./weights'))[1]:
        for j in next(os.walk(f'./weights/{i}'))[1]:
            models.append(os.path.join(os.getcwd(), 'weights', i, j))
    return models

def log_training(model_path, selecte_as_prior_weights, use_base_model_checkbox):

    config = {}
    with open(os.path.join(os.getcwd(), 'train.config.yml'), 'r') as stream:
        config=yaml.safe_load(stream)
        print(config)

    with open(os.path.join(os.getcwd(), 'concepts_list.json'), 'r') as json_file:
        concepts_list = json.load(json_file)
    
    with open(os.path.join(os.getcwd(), 'train.log.yml'), 'r') as stream:
        logs=yaml.safe_load(stream)


    if selecte_as_prior_weights != None and use_base_model_checkbox != True:
        for log in logs: 
            if log['weights_path'] == selecte_as_prior_weights:
                concepts_list = concepts_list + log['concepts_list']

    logs.append({
    "weights_path": os.path.join(os.getcwd(), 'weights', model_path, str(config['max_train_steps'])),
    "concepts_list": concepts_list
    })

    with open(os.path.join(os.getcwd(), 'train.log.yml'), 'w') as config:
        config.write(yaml.dump(logs))

def get_concepts(weights_path):
    concepts = []
    logs = {}
    with open(os.path.join(os.getcwd(), 'train.log.yml'), 'r') as stream:
        logs=yaml.safe_load(stream)

    for train_run in logs:
        print(f"train run paths: {train_run['weights_path']}  == {weights_path}")
        if train_run['weights_path'] == weights_path:
            concepts = train_run['concepts_list']
            #print(concepts)
            break
    
    concepts_to_return = []
    for concept in concepts:
        images = []
        os_walk = os.walk(concept['instance_data_dir'])
    
        for directory in os_walk:
            files = directory[2]
            if len(files) > 0:
                for file_path in files:
                    images.append(Image.open(os.path.join(concept['instance_data_dir'], file_path)))
            
        #for image_path in next(os.walk(concept['instance_data_dir']))[2]:
        #    print(image_path)
        #    images.append(
        #        Image.open(os.path.join(concept['instance_data_dir'], image_path))
        #    )
        
        concepts_to_return.append( {"instance_prompt": concept['instance_prompt'], "class_prompt": concept['class_prompt'], "instance_images": images} )
    print(concepts_to_return)
    return concepts_to_return