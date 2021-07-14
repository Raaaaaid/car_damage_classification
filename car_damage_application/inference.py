import tensorflow as tf
from tensorflow import keras

damage_names = ["minor", "moderate", "severe", "whole"]

### Expect tf decoded jpeg, already preprocessed
def inference(image, model, threshold, labels):
    result_dict = dict()
    result = model.predict(image)
    for i in range(len(result)):
        tensor = result[i]
        class_list = []
        for j in range(len(tensor)):
            if tensor[j] > threshold:
                class_list.append(labels[j])
        result_dict.update({i : class_list})
    return result_dict


if __name__ == "__main__":
    model = keras.models.load_model('C:/saved_models/ResNet50/resnet50_damage')
    image = tf.io.read_file('C:/Users/Pascal/Pictures/car_testing/car_mild.jpg')
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (150, 150))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, 0)
    result = inference(image, model, 0.7, damage_names)
    print(result)