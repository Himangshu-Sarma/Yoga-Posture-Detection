# BACKEND SERVER MADE USING FLASK 

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = Flask(__name__)
CORS(app)


MODEL = tf.keras.models.load_model('./Model_1.keras')
CLASS_NAMES = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']
TARGET_SIZE = (75, 75)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ping', methods=['GET'])
def ping():
    return "Hello, I am alive"



def read_file_as_image(data) -> np.ndarray:
    # image = np.array(Image.open(BytesIO(data)))
    # return image
    image = Image.open(BytesIO(data)).convert('RGB')
    image = image.resize(TARGET_SIZE)  # Resize the image to the target size
    image = np.array(image) 
    return image


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    try:
        image = read_file_as_image(file.read())
        img_batch = np.expand_dims(image, 0)

        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        # return jsonify({
            # 'class': predicted_class,
            # 'confidence': float(confidence),
            # 'predictions': str(predictions)
        # })
        return render_template('index.html', clas='{}'.format(predicted_class), confidence='{}'.format(confidence))
        
    except Exception as error:
        return str(error)
    
    # finally:
    #     file.close()

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)








#TEST SERVER(not working) MADE WITH FASTAPI

# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
# import os

# # print(os.getcwd())
# # print(os.listdir('../my_model.h5'))

# app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# MODEL = tf.keras.models.load_model('../Model_1.keras')


# CLASS_NAMES = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']


# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive"


# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image


# @app.post("/predict")
# async def predict(
#         file: UploadFile = File(...)
# ):
#     new_file = file.read()
#     image = read_file_as_image(new_file)
#     img_batch = np.expand_dims(image, 0)

#     predictions = MODEL.predict(img_batch)

#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
#     return {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }


# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)



