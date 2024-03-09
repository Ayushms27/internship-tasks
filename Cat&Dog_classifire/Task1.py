import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'C:/Users/AYUSHMS/Desktop/Bharat_Interns/train_data_dir',
        target_size=(150, 150),
        batch_size=2,
        class_mode='binary')

# Model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Model training
model.fit(train_generator, epochs=10)

# Save the model
model.save('cat_dog_classifier.keras')


from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('cat_dog_classifier.keras')

# Now you can use the loaded model for prediction or other tasks

import numpy as np
from tensorflow.keras.preprocessing import image

# Load an image for prediction
img_path = "C:/Users/AYUSHMS/Desktop/Bharat_Interns/exp2_img.jpg"
img = image.load_img(img_path, target_size=(150, 150))

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Add a dimension to the image array to match the input shape of the model
img_array = np.expand_dims(img_array, axis=0)

# Make a prediction on the new image
prediction = model.predict(img_array)

# Print the prediction result
print('Prediction:', prediction[0][0])