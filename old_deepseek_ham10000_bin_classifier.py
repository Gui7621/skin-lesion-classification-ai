import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Constants
IMAGE_SIZE = (224, 224)  # Standard size for many CNN architectures
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 1  # Binary classification
DATA_DIR = "HAM10000_images"  # Directory containing the images
METADATA_PATH = "HAM10000_metadata.csv"  # Path to metadata file

# Load and preprocess metadata
def load_and_filter_metadata(metadata_path, data_dir):
    """Load metadata and filter for only 'mel' and 'nv' classes, balancing the dataset."""
    # Load metadata
    df = pd.read_csv(metadata_path)
    
    # Filter for only 'mel' and 'nv' classes
    df = df[df['dx'].isin(['mel', 'nv'])]
    
    # Balance the dataset by sampling the same number of images from each class
    n_samples = min(len(df[df['dx'] == 'mel']), len(df[df['dx'] == 'nv']))
    df_mel = df[df['dx'] == 'mel'].sample(n_samples, random_state=seed)
    df_nv = df[df['dx'] == 'nv'].sample(n_samples, random_state=seed)
    df_balanced = pd.concat([df_mel, df_nv])
    
    # Create full image paths
    df_balanced['path'] = df_balanced['image_id'].apply(lambda x: os.path.join(data_dir, f"{x}.jpg"))
    
    # Convert labels to binary (mel=1, nv=0)
    df_balanced['label'] = df_balanced['dx'].apply(lambda x: 'mel' if x == 'mel' else 'nv')
    
    return df_balanced

# Load and filter the metadata
df = load_and_filter_metadata(METADATA_PATH, DATA_DIR)

# Split data into train (70%), validation (15%), and test (15%) sets
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=seed)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=seed)

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation and test

# Create data generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['nv', 'mel'],
    shuffle=True,
    seed=seed
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='path',
    y_col='label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['nv', 'mel'],
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='path',
    y_col='label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['nv', 'mel'],
    shuffle=False
)

# Model architecture (using EfficientNetB0 as base)
def create_model(input_shape=IMAGE_SIZE + (3,)):
    """Create a CNN model for binary classification."""
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model

# Create and compile the model
model = create_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_loss',
        save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3
    )
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=callbacks
)

# Evaluate on test set
print("\nEvaluating on test set...")
test_results = model.evaluate(test_generator)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")
print(f"Test AUC: {test_results[2]:.4f}")

# Gererate prediction to the confusion matrix
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = (y_pred_probs > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['nv (0)', 'mel (1)'], 
            yticklabels=['nv (0)', 'mel (1)'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

# Save the final model
model.save('old_melanoma_classifier.h5')
print("Model saved as 'melanoma_classifier.h5'")