import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.applications import VGG16
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from google.colab import drive
drive.mount('/content/drive')

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1  # Increased epochs for better training
WEIGHT_DECAY = 1e-5
LEARNING_RATE = 0.001
TRAIN_DIR = ('/content/drive/My Drive/archive/chest_xray/train')
VAL_DIR =  ('/content/drive/My Drive/archive/chest_xray/val')
TEST_DIR =  ('/content/drive/My Drive/archive/chest_xray/test')

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                                                    class_mode='binary')
val_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                                                class_mode='binary')
test_generator = test_datagen.flow_from_directory(TEST_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                                                  class_mode='binary')

# Build the CNN model with transfer learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pre-trained layers

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(WEIGHT_DECAY)),  # Apply weight decay using regularizer
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Add the l2 regularizer to each layer in the model
regularizer = tf.keras.regularizers.l2(1e-4)
for layer in model.layers:
    if hasattr(layer, 'kernel_regularizer'):
        layer.kernel_regularizer = regularizer
    if hasattr(layer, 'bias_regularizer'):
        layer.bias_regularizer = regularizer

model.compile(optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),  # Use Adam optimizer with decay
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")

model.save('/content/drive/My Drive/PneumoniaDetectionGenAI.h5')

test_images, test_labels = next(test_generator)
for i, image in enumerate(test_images):
    img_pil = tf.keras.preprocessing.image.array_to_img(image)
    img_pil.save(f"/content/drive/My Drive/PreprocessedImages/image_{i}.png")

