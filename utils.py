import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn import preprocessing


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
    le - sklearn.preprocessing.OneHotEncoder or sklearn.preprocessing.MultiLabelBinarizer:
        encoder to encode labels (strings) to a binary representation
        (multi-class: OneHotEncoder, multi-label: MultiLabelBinarizer)
    labels - list:
        list of labels as strings
    ---------------
    Outputs
    ---------------
    numpy.ndarray:
        list of labels as binary representations
'''
def encode_classes(le, labels):
    le.fit(labels)
    return le.transform(labels)

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
    Generate tf.dataset based on images and labels. 
    Lists have to be formatted depending on multi label or single label classification.
    Encoding labels to integer values [0, ...., num_classes].
    Does train, val splitting. Shuffling is important.
    Calls data augmentation and batching functions.
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
    augmented_train - tf.data.Dataset:
        image dataset the model will be trained on
    val - tf.data.Dataset:
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
    dataset = dataset.map(read_images).shuffle(len(images), seed=SEED)
    val = dataset.take(round(len(images) * valsplit)).map(unification).batch(8)
    train = dataset.skip(round(len(images) * valsplit))
    augmented_train = train
    for image, label in train:
        augmented_train = augmented_train.concatenate(augmentation(image, label))
    augmented_train = augmented_train.shuffle(len(images), seed=SEED).map(unification).batch(8)
    return augmented_train, val
