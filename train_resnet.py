import tensorflow as tf
import pandas as pd
import json
import datetime

from tensorflow import keras
from utils import load_dataset, generate_dataset

'''
    Builds a neural network on top of ResNet50V2. A different architecture is build depending
    on the labels (multi-class or multi-label case). The function implements early stopping,
    a slowly decreasing learning rate and a callback for tensorboard to visualize results.
    Tensorboard logs and the model itself are saved at model_path.
    Inputs
    ---------------
    images - list:
        list of full image filepaths
    labels - list:
        list of labels corresponding to the images
    num_classes - int:
       number of different classes 
    multilabel_flag - bool:
        True/False flag signalling multi label classification
    model_path - str:
        path where the model will be saved
    ---------------
    Outputs
    ---------------
    -
    ---------------
'''
def train_resnet(images, labels, num_classes, multilabel_flag, model_path):
    
    train, val = generate_dataset(images, labels, 0.15, multilabel_flag)

    # setting folder to save logs and model
    if multilabel_flag:
        if num_classes == 4:
            folder = "resnet50_location/"
        else:
            folder = "resnet50_damagetype/"
    else:
        folder = "resnet50_damage/"

    # set correct activation function, metric and loss function for multi-label and multi-class case
    if multilabel_flag:
        activation_func = "sigmoid"
        metric = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
        loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    else:
        activation_func = "softmax"
        metric = tf.keras.metrics.CategoricalAccuracy()
        loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    # Resnet50 Model Customization
    resnet50 = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=(150, 150, 3),
        pooling=None,
        classes=num_classes,
    )
    resnet50.trainable = False
    prediction_layer = tf.keras.layers.Dense(num_classes, activation=activation_func)
    model = tf.keras.Sequential([
        resnet50,
        tf.keras.layers.Flatten(),
        prediction_layer
    ])
    model.summary()

    # initialize early stopping callback
    if multilabel_flag:
        monitoring = "val_binary_accuracy"
    else:
        monitoring = "val_categorical_accuracy"
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitoring,
        patience=5,
        min_delta=0.001,
        mode="max"
    )

    # initialize a callback to decrease the learning rate over time
    def scheduler(epoch, learning_rate):
        if epoch < 6:
            return learning_rate
        else:
            return learning_rate * tf.math.exp(-0.1)
    decrease_learning_rate = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

    # initialize tensorboard for visualizing results
    log_dir = model_path + folder + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Training start
    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=loss_func,
        metrics=[metric])
    model.fit(train, epochs=500, batch_size=8, validation_data=val, verbose=1,
              callbacks=[decrease_learning_rate, tensorboard, early_stopping])

    # save model
    model.save(model_path + folder)


if __name__ == "__main__":

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    # read paths from training_paths.json
    with open("training_paths.json") as file:
        json_data = json.load(file)
    train_dataset_path = json_data["train_dataset_path"]
    csv_data_path = json_data["csv_data_path"]
    model_path = json_data["saving_model_path"]
    print("Training dataset at {}".format(train_dataset_path))
    print("*.csv file with labels at {}".format(csv_data_path))
    print("Trained models will be saved at {}".format(model_path))
    model_path += "saved_models/ResNet50/"

    # load dataset
    images, damage, location, damagetype = load_dataset(train_dataset_path, csv_data_path)
    print("Image count: {}".format(len(images)))

    # train nets
    train_resnet(images, damage, 4, False, model_path)
    #train_resnet(images, location, 4, True, model_path)
    #train_resnet(images, damagetype, 5, True, model_path)
