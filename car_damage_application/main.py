import json
import tensorflow as tf
from tensorflow import keras
from inference import inference

damage_model = None
damage_labels = None
damage_thresh = None

location_model = None
location_labels = None
location_thresh = None

type_model = None
type_labels = None
type_thresh = None

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

    with open(config_fp) as config_file:
        config = json.load(config_file)
    
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

def run(image_fp):
    global damage_model
    global damage_labels
    global damage_thresh
    global location_model
    global location_labels
    global location_thresh
    global type_model
    global type_labels
    global type_thresh

    image = tf.io.read_file(image_fp)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (150, 150))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, 0)

    damage = inference(image, damage_model, damage_thresh, damage_labels)
    location = inference(image, location_model, location_thresh, location_labels)
    damagetype = inference(image, type_model, type_thresh, type_labels)
    return (damage, location, damagetype)

if __name__ == "__main__":
    init("./application_config.json")
    img_fp = str(input("Image filepath: "))
    results = run(img_fp)
    print(results)