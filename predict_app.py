import base64
import numpy as np 
import io
from PIL import Image, ImageDraw
import keras
from keras import backend as K 
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import Flask, request, jsonify
import os
import tensorflow as tf
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PEOPLE_FOLDER = os.path.join('static', 'people_photo')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

def get_model():
    global model,graph
    model = load_model('models/40.h5')
    model.load_weights('models/40.w5')
    graph = tf.get_default_graph()
    print("Model loaded")
def preprocess_image(image,target_size):
    if image.mode!="RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image/=255.0
    image = np.expand_dims(image, axis = 0) 
    return image


print("Loading the model......")
get_model() 


@app.route("/predict",methods=["POST"])
def predict():

    random = np.random.randint(100000)+1
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image,target_size=(640,480))
    with graph.as_default():
        predict = model.predict(processed_image)
        axs = plt.imshow(image)
        rect = Rectangle((int(predict[0][0][0]),int(predict[3][0][0])),(int(predict[1][0][0])-int(predict[0][0][0])),(int(predict[2][0][0])-int(predict[3][0][0])),linewidth=1,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        bounded_image = axs.axes.add_patch(rect)
        plt.savefig('static/'+ str(random) +'.jpeg')
        
        response = {
        'prediction':{
        'x1' : int(predict[0][0][0]),
        'x2' : int(predict[1][0][0]),
        'y1' : int(predict[2][0][0]),
        'y2' : int(predict[3][0][0]),
        'name1' : int(random)
        }
        }
        rect.set_visible(False)
        
    return jsonify(response)



if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port)
