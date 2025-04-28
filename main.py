from google.colab import drive
drive.mount('/content/drive')
from google.colab import drive
drive.mount('/content/drive')
import zipfile

zip_path = '/content/drive/MyDrive/archive.zip'  # path in Drive
extract_dir = '/content'  # Extract to Colab's working directory

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extraction complete!")
import os

train_dir = '/content/pest/train'
test_dir = '/content/pest/test'

# List contents to verify
print("Train Directory:", os.listdir(train_dir))
print("Test Directory:", os.listdir(test_dir))
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_dir = '/content/pest/train'
test_dir = '/content/pest/test'
image_size = (128, 128)
batch_size = 16

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3)) #Loads a pre-trained model trained on ImageNet for image classification
base_model.trainable = False #Freezes the VGG16 layers so the pre-trained knowledge is not changed.

model = Model(inputs=base_model.input, outputs=Dense(train_generator.num_classes, activation='softmax')(GlobalAveragePooling2D()(base_model.output)))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_generator,
    epochs=2,
    validation_data=test_generator
)
model.save('/content/pest_cnn_model.h5')
print("✅ Model saved as 'pest_cnn_model.h5'")
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('/content/pest_cnn_model.h5')

# Define class labels based on your training set
class_labels = {
    0: 'mites',
    1: 'stem_borer',
    2: 'mosquito',
    3: 'grasshopper',
    4: 'aphids',
    5: 'sawfly',
    6: 'bollworm',
    7: 'armyworm',
    8: 'beetle'
}

# Replace the labels in the dictionary according to your dataset's actual labels

# Function to predict the class of a new image[]
def predict_pest(image_path):
    # Load and preprocess the image
    # Changed target_size to (128, 128) to match the model's input shape
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale the image as done during training

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the index of the highest probability
    predicted_label = class_labels[predicted_class]  # Map the index to the pest label

    return predicted_label

# Example usage:
image_path = '/content/pest/test/mosquito/jpg_11.jpg'  # Replace with the path to the new image you want to predict
predicted_pest = predict_pest(image_path)
print(f"Predicted Pest: {predicted_pest}")
model.save('/content/pest_cnn_model.h5')
model = tf.keras.models.load_model('/content/pest_cnn_model.h5')
def predict_with_confidence(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)[0]

    for i, prob in enumerate(predictions):
        print(f"{list(class_labels.values())[i]}: {prob*100:.2f}%")

predict_with_confidence('/content/pest/test/mites/jpg_0.jpg')
def predict_with_confidence(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)[0]

    for i, prob in enumerate(predictions):
        print(f"{list(class_labels.values())[i]}: {prob*100:.2f}%")

# Example usage:
predict_with_confidence('/content/pest/test/mosquito/jpg_11.jpg')
