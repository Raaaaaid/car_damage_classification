import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing

from utils import load_dataset, encode_classes, unification, read_images

'''
Script to evaluate used models on test dataset which can be found on kaggle. Link to the dataset is provided in the readme.
Uses a combination of functions already used in training and during the car damage application to load dataset and run it on provided models with the keras eval() function.
Accuracies are reported in the README.
'''
if __name__ == "__main__":
    with open('./application_config.json') as config_file:
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

    images, damage, location, damagetype = load_dataset('C:/Users/Pascal/Pictures/car_testing/test_data/', 'C:/Users/Pascal/Pictures/car_testing/test_data.csv')
    
    # Multi label Models
    ## Location
    location_labels_le = preprocessing.MultiLabelBinarizer()
    for i in range(len(location)):
        location[i] = location[i].strip('][').split(', ')
    loc_labels = encode_classes(location_labels_le, location)

    loc_dataset = tf.data.Dataset.from_tensor_slices((images, loc_labels)).map(read_images).map(unification).batch(8)
    print('Calculating location accuracy...')
    loc_results = location_model.evaluate(loc_dataset, batch_size=8)
    print(loc_results)
    print('Done.')

    ## Damage type
    type_labels_le = preprocessing.MultiLabelBinarizer()
    for i in range(len(damagetype)):
        damagetype[i] = damagetype[i].strip('][').split(', ')
    dt_labels = encode_classes(type_labels_le, damagetype)

    dt_dataset = tf.data.Dataset.from_tensor_slices((images, dt_labels)).map(read_images).map(unification).batch(8)
    print('Calculation damagetype accuracy...')
    type_results = type_model.evaluate(dt_dataset, batch_size=8)
    print(type_results)
    print('Done.')

    # Single label models
    ## Damage severity
    damage_labels_le = preprocessing.OneHotEncoder(sparse=False)
    dmg_labels = np.array(damage).reshape(-1, 1)
    dmg_labels = encode_classes(damage_labels_le, dmg_labels)

    dmg_dataset = tf.data.Dataset.from_tensor_slices((images, dmg_labels)).map(read_images).map(unification).batch(8)
    print('Calculating damage severity accuracy...')
    dmg_results = damage_model.evaluate(dmg_dataset, batch_size=8)
    print(dmg_results)
    print('Done.')