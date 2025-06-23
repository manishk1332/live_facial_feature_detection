import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense
from keras.applications import VGG19
from keras.utils import load_img, img_to_array
from keras.callbacks import ModelCheckpoint

# Configuration
IMG_WIDTH, IMG_HEIGHT = 48, 48
GENDER_IMG_WIDTH, GENDER_IMG_HEIGHT = 64, 64
BATCH_SIZE = 64
INITIAL_EPOCHS = 15 # Epochs for training the new classifier head
FINETUNE_EPOCHS = 10 # Epochs for fine-tuning the whole model
DATASET_PATH = 'datasets'
MODEL_DIR = 'models_trained_vgg19'

# Learning Rates
INITIAL_LR = 1e-4
FINETUNE_LR = 1e-5

# Create model directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Emotion Model Training using VGG19 Fine-Tuning

def create_emotion_model_vgg19(input_shape, num_classes):
    """Builds a fine-tunable emotion model based on VGG19."""
    # VGG19 expects 3-channel input. We will handle this in the data loading.
    # The input shape here must be (width, height, 3)
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model layers
    base_model.trainable = False

    # Add a custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def load_fer2013_from_folders():
    """Loads FER2013 dataset from a train/test directory structure for VGG19."""
    print("[INFO] Loading FER2013 dataset from folders...")
    
    fer_base_path = os.path.join(DATASET_PATH, 'fer2013')
    train_dir = os.path.join(fer_base_path, 'train')
    test_dir = os.path.join(fer_base_path, 'test')

    # Load images in RGB mode for VGG19
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        color_mode='rgb', 
        batch_size=BATCH_SIZE
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        color_mode='rgb',
        batch_size=BATCH_SIZE
    )
    
    class_names = train_dataset.class_names
    print(f"[INFO] Found emotion classes: {class_names}")

    # Pre-processing for VGG19
    train_dataset = train_dataset.map(lambda x, y: (tf.keras.applications.vgg19.preprocess_input(x), y))
    validation_dataset = validation_dataset.map(lambda x, y: (tf.keras.applications.vgg19.preprocess_input(x), y))

    return train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE), \
           validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE), \
           len(class_names)

def train_emotion_model():
    """Main function to fine-tune VGG19 for emotion."""
    train_ds, val_ds, num_classes = load_fer2013_from_folders()
    
    model, base_model = create_emotion_model_vgg19((IMG_WIDTH, IMG_HEIGHT, 3), num_classes)
    
    # --- Phase 1: Train the head ---
    print("[INFO] Phase 1: Training the custom classifier head...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR), loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint_path = os.path.join(MODEL_DIR, 'emotion_model_vgg19.h5')
    model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max')
    
    history = model.fit(
        train_ds,
        epochs=INITIAL_EPOCHS,
        validation_data=val_ds,
        callbacks=[model_checkpoint]
    )
    
    # Fine-tune the top layers
    print("\n[INFO] Phase 2: Fine-tuning the top VGG19 layers...")
    base_model.trainable = True

    # Unfreeze from this layer onwards
    fine_tune_at = 15 # Unfreeze from 'block4_conv1' onwards
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Re-compile with a very low learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR), loss='categorical_crossentropy', metrics=['accuracy'])

    total_epochs = INITIAL_EPOCHS + FINETUNE_EPOCHS
    initial_epoch_for_finetuning = INITIAL_EPOCHS
    print(f"[INFO] Starting fine-tuning from epoch {initial_epoch_for_finetuning}...")
    model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=initial_epoch_for_finetuning,
        validation_data=val_ds,
        callbacks=[model_checkpoint] # Continue saving the best model
    )
    print("[SUCCESS] Emotion model fine-tuning complete and saved.")

# Main Execution
if __name__ == "__main__":
    print("--- Starting Emotion Model Fine-Tuning ---")
    train_emotion_model()
    print(f"\n[COMPLETE] All models have been trained and saved to the '{MODEL_DIR}' directory.")