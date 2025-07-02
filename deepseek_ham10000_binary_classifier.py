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
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Constants
IMAGE_SIZE = (224, 224)  # EfficientNet default size
BATCH_SIZE = 32
EPOCHS = 10  # epochs
INIT_LR = 1e-4  # Lower learning rate
DATA_DIR = "HAM10000_images"
METADATA_PATH = "HAM10000_metadata.csv"

# Load and preprocess metadata
def load_and_filter_metadata(metadata_path, data_dir):
    """Load metadata and filter for only 'mel' and 'nv' classes."""
    df = pd.read_csv(metadata_path)
    df = df[df['dx'].isin(['mel', 'nv'])]
    
    # Balance classes
    n_samples = min(len(df[df['dx'] == 'mel']), len(df[df['dx'] == 'nv']))
    df_mel = df[df['dx'] == 'mel'].sample(n_samples, random_state=SEED)
    df_nv = df[df['dx'] == 'nv'].sample(n_samples, random_state=SEED)
    df_balanced = pd.concat([df_mel, df_nv])
    
    # Create paths and string labels
    df_balanced['path'] = df_balanced['image_id'].apply(lambda x: os.path.join(data_dir, f"{x}.jpg"))
    df_balanced['label'] = df_balanced['dx']  # Use string labels ('mel', 'nv')
    
    return df_balanced

# Load data
df = load_and_filter_metadata(METADATA_PATH, DATA_DIR)

# Train/Val/Test split (70/15/15)
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=SEED)

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./1)

val_test_datagen = ImageDataGenerator(rescale=1./1)

# Data generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['nv', 'mel'],  # nv=0, mel=1
    shuffle=True,
    seed=SEED
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

# Verify data loading
print("\nClass distributions:")
print(f"Train: {np.mean(train_generator.classes):.2f} (mel)")
print(f"Val: {np.mean(val_generator.classes):.2f} (mel)")
print(f"Test: {np.mean(test_generator.classes):.2f} (mel)")

# Model architecture (Unfrozen EfficientNet)
def create_model():
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=IMAGE_SIZE + (3,)
    )
    base_model.trainable = True  # Unfreeze all layers
    
    inputs = keras.Input(shape=IMAGE_SIZE + (3,))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    return model

# Create and compile model
model = create_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INIT_LR),
    loss='binary_crossentropy',
    metrics=['accuracy', 
             keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_auc',
        mode='max'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5
    )
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=EPOCHS,
    callbacks=callbacks
)

# Evaluation
print("\nEvaluating on test set...")
test_results = model.evaluate(test_generator)

test_loss = test_results[0]
test_acc = test_results[1]
test_auc = test_results[2]
test_precision = test_results[3]
test_recall = test_results[4]

print(f'\nTest Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test AUC: {test_auc:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')

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
    plt.figure(figsize=(12, 8))
    metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        plt.plot(history.history[metric], label='Train')
        plt.plot(history.history[f'val_{metric}'], label='Val')
        plt.title(metric)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_history(history)

# Save final model
model.save('melanoma_classifier_fixed.h5')
print("Model saved successfully!")