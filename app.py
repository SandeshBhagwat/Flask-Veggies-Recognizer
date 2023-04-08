from flask import Flask, render_template, request
import tensorflow_hub as hub
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
from keras.utils import img_to_array
from keras.utils import load_img
from werkzeug.utils import secure_filename
import os
import sys

class_map = ['Bean', 'Bitter Gourd', 'Bottle Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum',
             'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

app = Flask(__name__)

# Preparing and pre-processing the image


def load_and_prep_image(img_path):
    """
    Reads an image from filename, turns it into a tensor
    and reshapes it to (img_shape, img_shape, colour_channel).
    """
    # Read in target file (an image)
    img = tf.io.read_file(img_path)

    # Decode the read file into a tensor & ensure 3 colour channels
    # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
    img = tf.image.decode_image(img, channels=3)

    # Resize the image (to the same size our model was trained on)
    img = tf.image.resize(img, size=[224, 224])

    # Rescale the image (get all values between 0 and 1)
    img = img/255.
    # print(img)
    return img


def pred_and_plot(filename):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it
    img = load_and_prep_image(filename)
    print("predict")

    # Loading model
    model = load_model("/Users/sandeshbhagwat/Flask/envision-web-starter-code-main/Veggies_Classifier_CNN_model.h5",
                       custom_objects={'KerasLayer': hub.KerasLayer})

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    pred_class = class_map[int(tf.round(pred)[0][0])-1]
    # print(pred_class)
    return pred_class


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/prediction", methods=['GET', 'POST'])
def prediction_image():
    try:
      if request.method == "POST":
        img = request.files["file"]
        basepath = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(basepath, "uploads", secure_filename(img.filename))
        img.save(img_path)
        pred = pred_and_plot(img_path)
        # print(pred)
        return render_template("result.html", predictions=pred, img_path=img_path)
      
    except:
      error = "File not processed!"
      return render_template("result.html", err=error)


if __name__ == "__main__":
    app.run()
