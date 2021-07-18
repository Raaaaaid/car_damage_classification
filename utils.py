import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn import preprocessing, model_selection


SEED = 42


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
    imagepaths, damage_labels, location_labels, type_labels = list(), list(), list(), list()
    csv_data = pd.read_csv(csv_fp, dtype={'name': str, 'damage': str}, keep_default_na=True).fillna('[NULL]')
    csv_data['damagetype'] = csv_data['damagetype'].apply(lambda x: x.replace(";", ", "))
    csv_data['location'] = csv_data['location'].apply(lambda x: x.replace(";", ", "))
    for index, row in csv_data.iterrows():
        imagepaths.append(dataset_fp + str(row['name']))
    return imagepaths, csv_data['damage'].tolist(), csv_data['location'].tolist(), csv_data['damagetype'].tolist()

'''
    Encodes string labels to labels with a binary representation.
    Inputs
    ---------------
    encoder - sklearn.preprocessing.OneHotEncoder or sklearn.preprocessing.MultiLabelBinarizer:
        encoder to encode labels (strings) to a binary representation
        (multi-class: OneHotEncoder, multi-label: MultiLabelBinarizer)
        Translation:
            - damage labels: ["minor", "moderate", "severe", "whole"]
            - location labels: ["NULL", "front", "rear", "side"]
            - damagetype labels: ["NULL", "dent", "glass", "paint", "wheel"]
    labels - numpy.ndarray:
        list of labels as strings
    ---------------
    Outputs
    ---------------
    numpy.ndarray:
        list of labels as binary representations
'''
def encode_classes(encoder, labels):
    encoder.fit(labels)
    return encoder.transform(labels)

'''
    Reads an image file and decodes it to pixel values.
    Inputs
    ---------------
    image_file - tf.Tensor:
        tensor with the folderpath of the image as a string
    label - tf.Tensor:
        tensor with corresponding label
    ---------------
    Outputs:
    ---------------
    image - tf.Tensor:
        tensor with pixel values of the image
    label - tf.Tensor:
        tensor with corresponding label
'''
def read_images(image_file, label):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image, channels=3)
    return image, label

'''
    Resizes a given image to 150x150 and normalizes its values to [0, 1].
    Inputs
    ---------------
    image - tf.Tensor:
        tensor with pixel values of the image
    label - tf.Tensor:
        tensor with corresponding label
    ---------------
    Outputs
    ---------------
    tf.Tensor:
        tensor with resized and normalized pixel values of the image
    tf.Tensor:
        tensor with corresponding label
'''
def unification(image, label):
    image = tf.image.resize(image, (150, 150))
    return tf.cast(image, tf.float32) / 255.0, label

'''
    Transforms a given image. The operations are random flip up/down, flip left/right,
    randomly changed brightness and contrast.
    Inputs
    ---------------
    image - tf.Tensor:
        tensor with pixel values of the image
    label - tf.Tensor:
        tensor with corresponding label
    ---------------
    Outputs
    ---------------
    tf.data.Dataset:
        dataset with the augmented image and the corresponding label
'''
def augmentation(image, label):
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    image = tf.image.random_brightness(image, 0.2, seed=SEED)
    image = tf.image.random_contrast(image, 0.2, 0.5, seed=SEED)
    return tf.data.Dataset.from_tensors((image, label))

'''
    Generates cv_k train and validation tf.datasets based on images and labels. Splits are
    performed by a normal KFold (multi-label case) or a StratifiedKFold (multi-class case).
    Lists have to be formatted depending on multi label or single label classification. Also
    shuffles data, unificates images, encodes labels, calls data augmentation and batching
    functions.
    Inputs
    ---------------
    images - list:
        list of full image filepaths
    labels - list:
        list of labels corresponding to the images
    cv_k - int:
        number of folds for cross validation
    multilabel_flag - bool:
        True/False flag signalling multi label classification
    ---------------
    Yields
    ---------------
    augmented_train - tf.data.Dataset:
        image dataset the model will be trained on
    val - tf.data.Dataset:
        image dataset for model validation
'''
def generate_dataset(images, labels, cv_k, multilabel_flag):

    # convert lists to np.arrays
    images = np.array(images)
    labels = np.array(labels)

    # use normal KFold for multi-label case and StratifiedKFold for multi-class case
    if multilabel_flag:
        cv = model_selection.KFold(n_splits=cv_k, shuffle=True, random_state=SEED)
    else:
        cv = model_selection.StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=SEED)

    for train_ind, val_ind in cv.split(images, labels):
        train_images, train_labels = images[train_ind], labels[train_ind]
        val_images, val_labels = images[val_ind], labels[val_ind]

        # encode labels
        if multilabel_flag:
            encoder = preprocessing.MultiLabelBinarizer()
            def label_parser(label):
                return label.strip("][").split(", ")
            train_labels = np.array([label_parser(label) for label in train_labels])
            val_labels = np.array([label_parser(label) for label in val_labels])
        else:
            encoder = preprocessing.OneHotEncoder(sparse=False)
            train_labels = np.array(train_labels).reshape(-1, 1)
            val_labels = np.array(val_labels).reshape(-1, 1)
        train_labels, val_labels = encode_classes(encoder, train_labels), encode_classes(encoder, val_labels)

        # prepare validation images
        val = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).map(read_images)
        val = val.shuffle(len(val_images), seed=SEED).map(unification).batch(8)

        # prepare and augmentate training images
        train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).map(read_images)
        augmented_train = train
        for image, label in train:
            augmented_train = augmented_train.concatenate(augmentation(image, label))
        augmented_train = augmented_train.shuffle(len(augmented_train), seed=SEED).map(unification).batch(8)

        yield augmented_train, val
