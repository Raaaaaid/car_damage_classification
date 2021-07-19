import tensorflow as tf

'''
Loads given image and restructures it to fit model properties.
Includes resizing to model size, normalizing pixel values and expansion of dimensions.
    ---------------
    Inputs
    ---------------
    image_fp - string:
        file path to the image
    ---------------
    Outputs
    ---------------
    image - tensor:
        tf.image tensor ready for model input
'''
def preprocess_image(image):
    image = tf.image.resize(image, (150, 150))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, 0)
    return image

'''
    Doing single image inference with a given model.
    ---------------
    Inputs
    ---------------
    image - tf.image:
        normalized and resized 3-dim image tensor
    model - tf.keras model:
        detection model running the inference
    threshold - float:
        confidence threshold for result filtering
    labels - list:
        labels fitting to the keras model
    ---------------
    Outputs
    ---------------
    class_list - list:
        list of all detected labels for this image
'''
def single_inference(image, model, threshold, labels):
    result = model.predict(image)
    tensor = result[0]
    class_list = []
    for j in range(len(tensor)):
        if tensor[j] > threshold:
            class_list.append((labels[j], tensor[j]))
    return class_list