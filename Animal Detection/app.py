from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# load the saved model
model = tf.keras.models.load_model('animal_classifier.h5')

# define the class names
class_names = ['goat', 'sheep']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # retrieve the uploaded file
    file = request.files['file']

    # convert file to PIL image
    img = Image.open(file)

    # resize image
    img_resized = img.resize((244, 244))

    # convert image to numpy array
    img_array = np.array(img_resized)

    # normalize pixel values to be between 0 and 1
    img_array = img_array / 255.0

    # add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # make prediction
    prediction = model.predict(img_array)[0]

    # get class label with highest predicted probability
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]

    # create response object
    response = {
        'prediction': {
            'class': predicted_class,
            'probability': float(prediction[predicted_class_index])
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
