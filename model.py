import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.applications import VGG16, InceptionResNetV2
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax

train_dir = 'DATASET/TRAIN'
test_dir = 'DATASET/TEST'


train_datagen = ImageDataGenerator(width_shift_range= 0.1,
                                  horizontal_flip = True,
                                  rescale = 1./255,
                                  validation_split = 0.2)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                 validation_split = 0.2)



train_generator =  train_datagen.flow_from_directory(directory = train_dir,
                                                    target_size = (75,75),
                                                    color_mode = 'rgb',
                                                    class_mode = 'categorical',
                                                    batch_size = 16,
                                                    subset = 'training')

validation_generator  = test_datagen.flow_from_directory(directory = test_dir,
                                                  target_size = (75,75),
                                                  color_mode = 'rgb',
                                                  class_mode = 'categorical',
                                                  subset = 'validation')




model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding = 'Same', input_shape=(75, 75, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    #tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding = 'Same'),
    #tf.keras.layers.MaxPooling2D(2,2),
    #tf.keras.layers.Dropout(0.25),
    #tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding = 'Same'),
    #tf.keras.layers.MaxPooling2D(2,2),
    #tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding = 'Same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu',padding = 'Same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation='softmax')
])



optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer = optimizer,
              metrics=['accuracy'])
# epochs = 50  
# batch_size = 16


model.summary()


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


Result=model.fit(
    train_generator,
    epochs=20,
    batch_size=16,
    verbose=1,
    validation_data=validation_generator
)


test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(75, 75),
    color_mode='rgb',
    class_mode='categorical',  
    batch_size=16,
    subset=None 
)


IMAGE_SIZE=256
BATCH_SIZE=16
tr_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    "DATASET/TRAIN",
    shuffle=True,
     image_size=(IMAGE_SIZE,IMAGE_SIZE),
     batch_size=BATCH_SIZE
)


te_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    "DATASET/TEST",
    shuffle=True,
     image_size=(IMAGE_SIZE,IMAGE_SIZE),
     batch_size=BATCH_SIZE
)


class_name=tr_dataset.class_names
class_name



num_images=0
for data_batch, labels_batch in test_generator:
    first_image = (data_batch[0]*255).astype('uint8')
    first_label = labels_batch[0]

    print("First image to predict")
    plt.imshow(first_image)
    plt.show()
    print("Actual label:", class_name[np.argmax(first_label)])

    batch_prediction = model.predict(data_batch)
    print("Predicted label:", class_name[np.argmax(batch_prediction[0])])
    num_images += 1
    if num_images >= 20:
            break



#import tensorflow as tf
#import numpy as np

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Creating a batch with a single image

    predictions = model.predict(img_array)
    predicted_class = class_name[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    
    return predicted_class, confidence


#import matplotlib.pyplot as plt

plt.figure(figsize=(15, 15))

# Iterate over the generator to get batches of test data
for batch_images, batch_labels in test_generator:
    for i in range(min(batch_images.shape[0], 9)):  # Limit to 9 images per batch
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow((batch_images[i] * 255).astype('uint8'))
        
        # Assuming predict function returns predicted_class and confidence
        predicted_class, confidence = predict(model, batch_images[i])
        actual_class = class_name[np.argmax(batch_labels[i])]
        
        plt.title(f"Actual: {actual_class}\nPredicted: {predicted_class}\nConfidence: {confidence:.2f}%")
        plt.axis("off")
        
    break  # Stop after the first batch to show only 9 images

plt.tight_layout()
plt.show()


import pickle

model_version=1
model.save('D://YogaCNN/Model_1.keras')

pickle.dump(model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))