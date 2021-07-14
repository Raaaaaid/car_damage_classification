import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn import preprocessing

'''
Creates required label and image lists for further dataset preparation.
    Inputs
    ---------------
    dataset_fp - str: 
        path to root folder of image data including final dash
    csv_fp - str:
        path to csv file containing label data
    ---------------
    Outputs
    ---------------
    imagepaths - list:
        list containing all full image filepaths
    damage - list:
        list containing damage severity labels
    location - list:
        list containing damage location labels
    damagetype - list:
        list containing damage type labels
'''
def load_dataset(dataset_fp, csv_fp):
    imagepaths = list()
    csv_data = pd.read_csv(csv_fp, dtype={'name': str, 'damage': str}, keep_default_na=True).fillna('[NULL]')
    csv_data['damagetype'] = csv_data['damagetype'].apply(lambda x: x.replace(";", ", "))
    csv_data['location'] = csv_data['location'].apply(lambda x: x.replace(";", ", "))
    for index, row in csv_data.iterrows():
        imagepaths.append(dataset_fp + str(row['name']))
    return imagepaths, csv_data['damage'].tolist(), csv_data['location'].tolist(), csv_data['damagetype'].tolist()

def encode_classes(le, labels):
    le.fit(labels)
    return le.transform(labels)

def read_images(image_file, label):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image, channels=3)
    return image, label

def unification(image_file, label):
    image = tf.image.resize(image_file, (150, 150))
    return tf.cast(image, tf.float32) / 255.0, label

def augmentation(image_file, label):
    # Data augmentation here
    return image_file, label

'''
    Generate tf.dataset based on images and labels. 
    Lists have to be formatted depending on multi label or single label classification.
    Encoding labels to integer values [0, ...., num_classes].
    Does train, val splitting. Shuffling is important.
    Calls data augmentation and batching functions.
    ---------------
    Inputs
    ---------------
    images - list:
        list of full image filepaths
    labels - list:
        list of labels corresponding to the images
    valsplit - float:
        percentage of images used for validation dataset
    multilabel_flag - bool:
        True/False flag signalling multi label classification
    ---------------
    Outputs
    ---------------
    train - tf.dataset:
        image dataset the model will be trained on
    val - tf.dataset:
        image dataset for model validation
'''
def generate_dataset(images, labels, valsplit, multilabel_flag):
    if multilabel_flag:
        labels_le = preprocessing.MultiLabelBinarizer()
        for i in range(len(labels)):
            labels[i] = labels[i].strip('][').split(', ')
    else:
        labels_le = preprocessing.OneHotEncoder(sparse=False)
        labels = np.array(labels).reshape(-1, 1)
    labels = encode_classes(labels_le, labels)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(read_images).shuffle(len(images))
    val = dataset.take(round(len(images) * valsplit)).map(unification).batch(8)
    train = dataset.skip(round(len(images) * valsplit)).map(augmentation).map(unification).batch(8)
    return train, val