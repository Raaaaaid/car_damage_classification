import json
import tensorflow as tf
from tensorflow import keras
from .inference import single_inference, preprocess_image

damage_model = None
damage_labels = None
damage_thresh = None

location_model = None
location_labels = None
location_thresh = None

type_model = None
type_labels = None
type_thresh = None

debug = "False"

'''
Initialization of models, label lists and threshold values.
Parameters get read from a Json file (see application_config.json).
Since TensorFlow allocates resources on the first run of the model, we do this in the initialization step.
This nets a small time gain on the first detection of an actual image.
    ---------------
    Inputs
    ---------------
    config_fp - string:
        file path to configuration json
'''
def init(config_fp):
    global damage_model
    global damage_labels
    global damage_thresh
    global location_model
    global location_labels
    global location_thresh
    global type_model
    global type_labels
    global type_thresh
    global debug

    with open(config_fp) as config_file:
        config = json.load(config_file)
    
    debug = config['debug']

    damage_model = keras.models.load_model(config['damage_model_fp'])
    damage_thresh = config['damage_conf_threshold']
    with open(config['damage_label_fp']) as txt:
        damage_labels = txt.read().splitlines()
    
    location_model = keras.models.load_model(config['location_model_fp'])
    location_thresh = config['location_conf_threshold']
    with open(config['location_label_fp']) as txt:
        location_labels = txt.read().splitlines()

    type_model = keras.models.load_model(config['damagetype_model_fp'])
    type_thresh = config['damagetype_conf_threshold']
    with open(config['damagetype_label_fp']) as txt:
        type_labels = txt.read().splitlines()

    prep_image = tf.zeros([1, 150, 150, 3])
    damage_model.predict(prep_image)
    location_model.predict(prep_image)
    type_model.predict(prep_image)

'''
Main classification pipeline.
Takes an image and calls preprocessing, feeds the image into damage model.
Depending on damage model output feed image into location model and damagetype model respectively.
Fill dictionary with results.
    ---------------
    Inputs
    ---------------
    image_fp - string
        file path to the image
    ---------------
    Outputs
    ---------------
    result_dict - dict
        dictionary containing car damage classifications in form {labeltype : [(class, confidence)]}
'''
def run_classification(image):
    global damage_model
    global damage_labels
    global damage_thresh
    global location_model
    global location_labels
    global location_thresh
    global type_model
    global type_labels
    global type_thresh

    result_dict = dict()
    try:
        image = preprocess_image(image)

        damage = single_inference(image, damage_model, damage_thresh, damage_labels)
        result_dict['damage'] = damage
        if damage[0][0] != "whole":
            location = single_inference(image, location_model, location_thresh, location_labels)
            result_dict['location'] = location
            damagetype = single_inference(image, type_model, type_thresh, type_labels)
            result_dict['damagetype'] = damagetype
    except Exception as e:
        print(type(e))
        print(e)
        pass
    return result_dict