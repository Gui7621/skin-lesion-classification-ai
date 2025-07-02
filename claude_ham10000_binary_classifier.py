#!/usr/bin/env python3
"""
HAM10000 Binary Image Classifier: Melanoma vs Nevus
This script trains a CNN to distinguish between melanoma (mel) and nevus (nv) 
from the HAM10000 dataset using TensorFlow/Keras.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
CONFIG = {
    'img_size': (224, 224),  # Standard size for EfficientNet
    'batch_size': 32,
    'epochs': 5,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'test_split': 0.1,
    'target_classes': ['mel', 'nv'],  # melanoma and nevus
    'data_dir': 'HAM10000_images',  # Directory containing images
    'metadata_file': 'HAM10000_metadata.csv'  # CSV with image metadata
}

def check_data_setup():
    """
    Check if data files and directories exist and show their structure.
    """
    print("Checking data setup...")
    print("=" * 40)
    
    # Check metadata file
    if os.path.exists(CONFIG['metadata_file']):
        print(f"✓ Metadata file found: {CONFIG['metadata_file']}")
        df = pd.read_csv(CONFIG['metadata_file'])
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
        if 'dx' in df.columns:
            print(f"  - Classes: {df['dx'].value_counts().to_dict()}")
        else:
            print(f"  - Warning: 'dx' column not found")
    else:
        print(f"✗ Metadata file not found: {CONFIG['metadata_file']}")
    
    # Check images directory
    if os.path.exists(CONFIG['data_dir']):
        print(f"✓ Images directory found: {CONFIG['data_dir']}")
        image_files = [f for f in os.listdir(CONFIG['data_dir']) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  - Image files found: {len(image_files)}")
        if image_files:
            print(f"  - Sample files: {image_files[:5]}")
    else:
        print(f"✗ Images directory not found: {CONFIG['data_dir']}")
    
    print("=" * 40)

def load_and_filter_metadata(metadata_path, target_classes):
    """
    Load metadata and filter for target classes only.
    """
    print("Loading and filtering metadata...")
    
    # Check if metadata file exists
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load the metadata CSV file
    df = pd.read_csv(metadata_path)
    
    # Display first few rows and columns to understand structure
    print(f"Metadata file columns: {list(df.columns)}")
    print(f"First 5 rows:")
    print(df.head())
    
    # Check if 'dx' column exists, if not, try common alternatives
    if 'dx' not in df.columns:
        possible_cols = ['diagnosis', 'label', 'class', 'category', 'type']
        dx_col = None
        for col in possible_cols:
            if col in df.columns:
                dx_col = col
                break
        
        if dx_col:
            print(f"Using column '{dx_col}' as diagnosis column")
            df = df.rename(columns={dx_col: 'dx'})
        else:
            raise ValueError(f"No diagnosis column found. Available columns: {list(df.columns)}")
    
    # Display unique values in dx column
    print(f"Unique values in diagnosis column: {df['dx'].unique()}")
    
    # Filter for only mel and nv classes
    df_filtered = df[df['dx'].isin(target_classes)].copy()
    
    print(f"Original dataset size: {len(df)}")
    print(f"Filtered dataset size: {len(df_filtered)}")
    
    if len(df_filtered) == 0:
        print(f"WARNING: No samples found for classes {target_classes}")
        print(f"Available classes: {df['dx'].value_counts()}")
        raise ValueError(f"No samples found for target classes {target_classes}")
    
    print(f"Class distribution:\n{df_filtered['dx'].value_counts()}")
    
    return df_filtered

def balance_dataset(df, target_classes):
    """
    Balance the dataset by undersampling the majority class.
    """
    print("\nBalancing dataset...")
    
    # Get class counts
    class_counts = df['dx'].value_counts()
    min_count = min(class_counts.values)
    
    # Sample equal numbers from each class
    balanced_dfs = []
    for class_name in target_classes:
        class_df = df[df['dx'] == class_name].sample(n=min_count, random_state=42)
        balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced dataset size: {len(balanced_df)}")
    print(f"Balanced class distribution:\n{balanced_df['dx'].value_counts()}")
    
    return balanced_df

def create_data_generators(df, data_dir, img_size, batch_size, validation_split, test_split):
    """
    Create data generators for train, validation, and test sets.
    """
    print("\nCreating data generators...")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Check if image_id column exists, try alternatives
    if 'image_id' not in df.columns:
        possible_cols = ['image', 'filename', 'file', 'image_name', 'id']
        img_col = None
        for col in possible_cols:
            if col in df.columns:
                img_col = col
                break
        
        if img_col:
            print(f"Using column '{img_col}' as image filename column")
            df = df.rename(columns={img_col: 'image_id'})
        else:
            raise ValueError(f"No image filename column found. Available columns: {list(df.columns)}")
    
    # Ensure image_id has proper extension
    if not df['image_id'].iloc[0].endswith(('.jpg', '.jpeg', '.png')):
        print("Adding .jpg extension to image filenames")
        df['image_id'] = df['image_id'] + '.jpg'
    
    # Check if some images actually exist
    sample_images = df['image_id'].head(5).tolist()
    existing_images = []
    for img in sample_images:
        img_path = os.path.join(data_dir, img)
        if os.path.exists(img_path):
            existing_images.append(img)
        else:
            print(f"Image not found: {img_path}")
    
    if len(existing_images) == 0:
        print(f"No images found in directory: {data_dir}")
        print(f"Directory contents: {os.listdir(data_dir)[:10]}")  # Show first 10 files
        raise FileNotFoundError("No matching images found in the specified directory")
    
    print(f"Found {len(existing_images)} out of {len(sample_images)} sample images")
    
    # Split data into train, validation, and test
    train_df, temp_df = train_test_split(df, test_size=validation_split + test_split, 
                                        random_state=42, stratify=df['dx'])
    val_df, test_df = train_test_split(temp_df, test_size=test_split/(validation_split + test_split), 
                                      random_state=42, stratify=temp_df['dx'])
    
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators with better error handling
    try:
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            directory=data_dir,
            x_col='image_id',
            y_col='dx',
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True,
            seed=42
        )
        
        if train_generator.samples == 0:
            raise ValueError("Training generator has 0 samples")
            
    except Exception as e:
        print(f"Error creating training generator: {e}")
        print("Checking data format...")
        print(f"Sample from train_df:\n{train_df.head()}")
        raise
    
    val_generator = val_test_datagen.flow_from_dataframe(
        val_df,
        directory=data_dir,
        x_col='image_id',
        y_col='dx',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_dataframe(
        test_df,
        directory=data_dir,
        x_col='image_id',
        y_col='dx',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"Generators created successfully:")
    print(f"  Train samples: {train_generator.samples}")
    print(f"  Validation samples: {val_generator.samples}")
    print(f"  Test samples: {test_generator.samples}")
    print(f"  Class indices: {train_generator.class_indices}")
    
    return train_generator, val_generator, test_generator, train_df, val_df, test_df

def create_model(img_size, learning_rate):
    """
    Create a CNN model using EfficientNetB0 as base with custom head.
    """
    print("\nCreating model...")
    
    # Load pre-trained EfficientNetB0 without top layers
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print(f"Model created with {model.count_params():,} parameters")
    return model

def train_model(model, train_gen, val_gen, epochs):
    """
    Train the model with callbacks for early stopping and learning rate reduction.
    """
    print("\nStarting model training...")
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Calculate steps per epoch
    steps_per_epoch = train_gen.samples // train_gen.batch_size
    validation_steps = val_gen.samples // val_gen.batch_size
    
    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def fine_tune_model(model, train_gen, val_gen, epochs=10):
    """
    Fine-tune the model by unfreezing some layers of the base model.
    """
    print("\nFine-tuning model...")
    
    # Unfreeze the top layers of the base model
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = len(base_model.layers) - 20
    
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Re-compile with a lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001/10),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Continue training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            verbose=1
        )
    ]
    
    steps_per_epoch = train_gen.samples // train_gen.batch_size
    validation_steps = val_gen.samples // val_gen.batch_size
    
    fine_tune_history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    return fine_tune_history

def evaluate_model(model, test_gen, test_df):
    """
    Evaluate the model on test data and generate classification report.
    """
    print("\nEvaluating model...")
    
    # Get predictions
    test_steps = test_gen.samples // test_gen.batch_size + 1
    predictions = model.predict(test_gen, steps=test_steps, verbose=1)
    
    # Convert predictions to binary
    y_pred = (predictions > 0.5).astype(int).flatten()
    
    # Get true labels
    test_gen.reset()
    y_true = test_gen.classes[:len(y_pred)]
    
    # Get class names
    class_names = list(test_gen.class_indices.keys())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Calculate and print accuracy
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    print(f"\nTest Accuracy: {accuracy:.4f}")

def plot_training_history(history, fine_tune_history=None):
    """
    Plot training history including loss and metrics.
    """
    plt.figure(figsize=(15, 5))
    
    # Combine histories if fine-tuning was performed
    if fine_tune_history:
        # Combine the histories
        total_epochs = len(history.history['loss']) + len(fine_tune_history.history['loss'])
        epochs_range = range(total_epochs)
        
        loss = history.history['loss'] + fine_tune_history.history['loss']
        val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']
        accuracy = history.history['accuracy'] + fine_tune_history.history['accuracy']
        val_accuracy = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
        
        # Mark where fine-tuning started
        fine_tune_start = len(history.history['loss'])
    else:
        epochs_range = range(len(history.history['loss']))
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        fine_tune_start = None
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    if fine_tune_start:
        plt.axvline(x=fine_tune_start, color='r', linestyle='--', alpha=0.7, label='Fine-tuning Start')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    if fine_tune_start:
        plt.axvline(x=fine_tune_start, color='r', linestyle='--', alpha=0.7, label='Fine-tuning Start')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to orchestrate the training process.
    """
    print("HAM10000 Binary Classifier: Melanoma vs Nevus")
    print("=" * 50)
    
    # Check if GPU is available
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    
    # First, check data setup
    check_data_setup()
    
    try:
        # Step 1: Load and prepare data
        df = load_and_filter_metadata(CONFIG['metadata_file'], CONFIG['target_classes'])
        balanced_df = balance_dataset(df, CONFIG['target_classes'])
        
        # Step 2: Create data generators
        train_gen, val_gen, test_gen, train_df, val_df, test_df = create_data_generators(
            balanced_df, 
            CONFIG['data_dir'], 
            CONFIG['img_size'], 
            CONFIG['batch_size'],
            CONFIG['validation_split'],
            CONFIG['test_split']
        )
        
        # Step 3: Create and train model
        model = create_model(CONFIG['img_size'], CONFIG['learning_rate'])
        
        # Initial training
        history = train_model(model, train_gen, val_gen, CONFIG['epochs'])
        
        # Step 4: Fine-tune the model
        fine_tune_history = fine_tune_model(model, train_gen, val_gen, epochs=10)
        
        # Step 5: Evaluate the model
        evaluate_model(model, test_gen, test_df)
        
        # Step 6: Plot training history
        plot_training_history(history, fine_tune_history)
        
        # Step 7: Save the final model
        model.save('ham10000_melanoma_classifier.h5')
        print("\nModel saved as 'ham10000_melanoma_classifier.h5'")
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the metadata CSV file exists and has the correct format")
        print("2. Check that the images directory exists and contains .jpg/.png files")
        print("3. Verify that image filenames in CSV match actual files")
        print("4. Ensure the CSV has columns for image filenames and diagnosis")
        print("5. Check that 'mel' and 'nv' classes exist in your data")
        
        # Show current directory contents for debugging
        print(f"\nCurrent directory contents:")
        print([f for f in os.listdir('.') if not f.startswith('.')])
        
        raise

if __name__ == "__main__":
    main()

# Additional utility functions for inference
def predict_single_image(model_path, image_path, img_size=(224, 224)):
    """
    Utility function to predict on a single image.
    """
    model = keras.models.load_model(model_path)
    
    # Load and preprocess image
    img = keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    
    # Convert to class label
    if prediction > 0.5:
        return "nevus", prediction
    else:
        return "melanoma", 1 - prediction

# Example usage for single image prediction:
# result, confidence = predict_single_image('ham10000_melanoma_classifier.h5', 'path_to_image.jpg')
# print(f"Prediction: {result} (confidence: {confidence:.4f})")