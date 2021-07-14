import tensorflow as tf
import pandas as pd

from tensorflow import keras
from utils import load_dataset, generate_dataset

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

def train_resnet(images, labeltype, num_classes, multilabel_flag):
    
    train, val = generate_dataset(images, labeltype, 0.15, multilabel_flag)

    # Resnet50 Model Customization for damage labels
    resnet50 = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=(150, 150, 3),
        pooling=None,
        classes=num_classes,
    )
    resnet50.trainable = False
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='sigmoid')
    model = tf.keras.Sequential([
        resnet50,
        tf.keras.layers.Flatten(),
        prediction_layer
    ])
    model.summary()

    ### Training start
    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['acc'])

    model.fit(train, epochs=30, batch_size=8, validation_data = val, verbose=1)

    return model

### Dataset exploration
TRAIN_DATASET_FP = "C:/Users/Pascal/Pictures/car-damage-dataset-new/data/"
CSV_DATA_FP = "C:/Users/Pascal/Pictures/car-damage-dataset-new/new_data.csv"

images, damage, location, damagetype = load_dataset(TRAIN_DATASET_FP, CSV_DATA_FP)
image_count = len(images)
print("Image count: {}".format(image_count))

#damage_model = train_resnet(images, damage, 4, False)
#damage_model.save("/saved_models/ResNet50/resnet50_damage")

location_model = train_resnet(images, location, 4, True)
location_model.save("/saved_models/ResNet50/resnet50_location")

damagetype_model = train_resnet(images, damagetype, 5, True)
damagetype_model.save("/saved_models/ResNet50/resnet50_damagetype")