### Virus - bacteria image split

import os
import shutil # not included in the report, mistake realised after submission

# Define directories
folders = ['chest_xray/train', 'chest_xray/val', 'chest_xray/test']

# Loop through each folder (train, validate, test)
for folder in folders:
    pneumonia_dir = os.path.join(folder, 'PNEUMONIA')  # Pneumonia folder
    virus_dir = os.path.join(folder, 'virus')          # New virus folder
    bacteria_dir = os.path.join(folder, 'bacteria')    # New bacteria folder

    # Create new folders for 'virus' and 'bacteria' if they don't exist
    os.makedirs(virus_dir, exist_ok=True)
    os.makedirs(bacteria_dir, exist_ok=True)

    # Move files based on the naming pattern
    for filename in os.listdir(pneumonia_dir):
        file_path = os.path.join(pneumonia_dir, filename)

        # Skip if not a file
        if not os.path.isfile(file_path):
            continue

        # Move virus images to the virus folder
        if 'virus' in filename.lower():
            shutil.move(file_path, os.path.join(virus_dir, filename))
            print(f"Moved {filename} to virus folder")

        # Move bacteria images to the bacteria folder
        elif 'bacteria' in filename.lower():
            shutil.move(file_path, os.path.join(bacteria_dir, filename))
            print(f"Moved {filename} to bacteria folder")

    # Remove the empty 'pneumonia' folder if all images have been moved
    if not os.listdir(pneumonia_dir):  # Check if folder is empty
        os.rmdir(pneumonia_dir)
        print(f"Removed empty pneumonia folder from {folder}")

print("Separation of virus and bacteria complete!")

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define directories
base_dir = './'
train_dir = os.path.join(base_dir, 'chest_xray/train')
val_dir = os.path.join(base_dir, 'chest_xray/val')
test_dir = os.path.join(base_dir, 'chest_xray/test')

# Image dimensions
img_height, img_width = 512, 512

# Set batch sizes
train_batch_size = 32
val_test_batch_size = 16

# Initialize the ImageDataGenerator with additional augmentations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Function to create a custom subset generator
def custom_subset_generator(generator, directory, classes, subset_size, batch_size):
    class_dirs = [os.path.join(directory, class_name) for class_name in classes]

    while True:
        selected_images = []
        for class_dir in class_dirs:
            images_in_class = os.listdir(class_dir)
            selected = random.sample(images_in_class, min(subset_size, len(images_in_class)))
            selected_images += [os.path.join(class_dir, img) for img in selected]

        random.shuffle(selected_images)

        for i in range(0, len(selected_images), batch_size):
            batch_images = selected_images[i:i+batch_size]
            batch_x, batch_y = [], []

            for img_path in batch_images:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                batch_x.append(img_array)

                class_name = os.path.basename(os.path.dirname(img_path))
                class_index = classes.index(class_name)
                batch_y.append(tf.keras.utils.to_categorical(class_index, num_classes=len(classes)))

            yield np.array(batch_x), np.array(batch_y)

# Classes (subfolder names)
classes = ['NORMAL', 'virus', 'bacteria']

# Create generators
train_generator = custom_subset_generator(train_datagen, train_dir, classes, subset_size=20, batch_size=train_batch_size)
val_generator = custom_subset_generator(val_datagen, val_dir, classes, subset_size=4, batch_size=val_test_batch_size)
test_generator = custom_subset_generator(test_datagen, test_dir, classes, subset_size=4, batch_size=val_test_batch_size)

# Calculate steps dynamically
steps_per_epoch = np.ceil(5221 / train_batch_size).astype(int)
validation_steps = np.ceil(16 / val_test_batch_size).astype(int)
test_steps = np.ceil(624 / val_test_batch_size).astype(int)

# Load ResNet50V2 with ImageNet weights
base_model = ResNet50V2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

# Fine-tune more layers by unfreezing fewer layers (only last 100 layers)
for layer in base_model.layers[:-100]:
    layer.trainable = False

# Model architecture with added Dropout and L2 regularization
model = tf.keras.models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

# Compile the model with a lower learning rate for fine-tuning
custom_learning_rate = 1e-5
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=custom_learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks with increased patience in EarlyStopping
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=50,
    callbacks=callbacks
)

# Plot training & validation accuracy and loss values
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_generator, steps=test_steps)
print(f"Test accuracy: {test_acc}")

# Collect all predictions and true labels
y_true = []
y_pred = []

for _ in range(test_steps):
    x_batch, y_batch = next(test_generator)
    predictions = model.predict(x_batch, verbose=0) # verbose=0 part is not in the report, it hides the steps under test accuracy.
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

# Classification report
class_labels = classes  # ['NORMAL', 'virus', 'bacteria']
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
