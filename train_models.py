import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, BatchNormalization, Dropout, Flatten, Dense
from keras.applications import VGG19
from keras.optimizers import Adam
from keras.utils import load_img, img_to_array
from keras.callbacks import ModelCheckpoint

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 48, 48
UTK_IMG_WIDTH, UTK_IMG_HEIGHT = 64, 64
BATCH_SIZE = 64
INITIAL_EPOCHS = 15 
FINETUNE_EPOCHS = 10 
DATASET_PATH = 'datasets'
MODEL_DIR = 'models_trained_vgg19_full'

# Learning Rates
INITIAL_LR = 1e-4
FINETUNE_LR = 1e-5

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- 1. Emotion Model Training ---
# For brevity, this is omitted. It should be the same as the previous correct version.
# If you need it, I can add it back.

# --- 2. UTKFace Models (Age, Gender, Race) ---

def create_vgg19_base_model(input_shape, trainable=False):
    """Creates a VGG19 base model."""
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = trainable
    return base_model

def build_classification_model(base_model, num_units, dropout, activation):
    """Builds a classification head for a base model."""
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_units, activation='relu')(x)
    x = Dropout(dropout)(x)
    predictions = Dense(1 if activation == 'sigmoid' else 5, activation=activation)(x)
    return Model(inputs=base_model.input, outputs=predictions)
    
def build_regression_model(base_model, num_units, dropout):
    """Builds a regression head for a base model."""
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_units, activation='relu')(x)
    x = Dropout(dropout)(x)
    predictions = Dense(1, activation='linear')(x)
    return Model(inputs=base_model.input, outputs=predictions)

def load_and_process_utkface():
    """
    Loads UTKFace and extracts age, gender, and race labels.
    FIX: Processes each image and its labels atomically to prevent list mismatches.
    """
    print("[INFO] Loading and preprocessing UTKFace dataset for all tasks...")
    images, ages, genders, races = [], [], [], []
    utk_dir = os.path.join(DATASET_PATH, 'UTKFace')

    if not os.path.isdir(utk_dir):
        print(f"[ERROR] Directory not found: {utk_dir}")
        exit()

    for filename in os.listdir(utk_dir):
        if filename.endswith('.jpg'):
            try:
                # 1. Parse labels from filename
                parts = filename.split('_')
                if len(parts) < 4:
                    continue # Skip malformed filenames
                
                age_val = int(parts[0])
                gender_val = int(parts[1])
                race_val = int(parts[2])

                # 2. Try to load the corresponding image
                image_path = os.path.join(utk_dir, filename)
                img = load_img(image_path, color_mode="rgb", target_size=(UTK_IMG_WIDTH, UTK_IMG_HEIGHT))
                
                # 3. If image loads successfully, add both image and labels to lists
                images.append(img_to_array(img))
                ages.append(age_val)
                genders.append(gender_val)
                races.append(race_val)

            except Exception as e:
                # This catches errors from parsing OR image loading
                print(f"[WARNING] Skipping file {filename} due to error: {e}")
                continue
    
    if not images:
        print(f"[ERROR] No valid images were loaded from: {utk_dir}")
        exit()
    
    # Convert lists to numpy arrays
    images = np.array(images, dtype='float32')
    ages = np.array(ages, dtype='float32')
    genders = np.array(genders, dtype='uint8')
    races = tf.keras.utils.to_categorical(races, num_classes=5)
    
    # Preprocess images for VGG19
    images = tf.keras.applications.vgg19.preprocess_input(images)
    
    print(f"[INFO] Successfully loaded {len(images)} images and labels.")
    return images, ages, genders, races

def train_model(X_train, y_train, X_test, y_test, model_type, model_name):
    """A generic function to train a model."""
    print(f"\n--- Training {model_name} Model ---")
    
    input_shape = (UTK_IMG_WIDTH, UTK_IMG_HEIGHT, 3)
    base_model = create_vgg19_base_model(input_shape, trainable=False)

    if model_type == 'gender':
        model = build_classification_model(base_model, 128, 0.5, 'sigmoid')
        loss, metric, monitor_metric, monitor_mode = 'binary_crossentropy', 'accuracy', 'val_accuracy', 'max'
    elif model_type == 'age':
        model = build_regression_model(base_model, 128, 0.4)
        loss, metric, monitor_metric, monitor_mode = 'mean_squared_error', 'mae', 'val_mae', 'min'
    elif model_type == 'race':
        model = build_classification_model(base_model, 256, 0.5, 'softmax')
        loss, metric, monitor_metric, monitor_mode = 'categorical_crossentropy', 'accuracy', 'val_accuracy', 'max'
    else: return

    # Phase 1
    print(f"[INFO] Phase 1: Training head for {model_name}...")
    model.compile(optimizer=Adam(learning_rate=INITIAL_LR), loss=loss, metrics=[metric])
    checkpoint_path = os.path.join(MODEL_DIR, f'{model_name}_model_vgg19.h5')
    model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor=monitor_metric, mode=monitor_mode)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=INITIAL_EPOCHS, validation_data=(X_test, y_test), callbacks=[model_checkpoint])

    # Phase 2
    print(f"\n[INFO] Phase 2: Fine-tuning for {model_name}...")
    base_model.trainable = True
    for layer in base_model.layers[:15]: layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=FINETUNE_LR), loss=loss, metrics=[metric])
    total_epochs = INITIAL_EPOCHS + FINETUNE_EPOCHS
    initial_epoch_for_finetuning = INITIAL_EPOCHS
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=total_epochs, initial_epoch=initial_epoch_for_finetuning, validation_data=(X_test, y_test), callbacks=[model_checkpoint])
    print(f"[SUCCESS] {model_name} model fine-tuning complete.")


# --- Main Execution ---
if __name__ == "__main__":
    # Load and split data for all UTKFace tasks at once
    images, ages, genders, races = load_and_process_utkface()
    
    # Split for Gender
    #X_train_gender, X_test_gender, y_train_gender, y_test_gender = train_test_split(images, genders, test_size=0.2, random_state=42)
    # Split for Age
    #X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(images, ages, test_size=0.2, random_state=42)
    # Split for Race
    X_train_race, X_test_race, y_train_race, y_test_race = train_test_split(images, races, test_size=0.2, random_state=42)

    # Train each model
    #train_model(X_train_gender, y_train_gender, X_test_gender, y_test_gender, 'gender', 'gender')
    #train_model(X_train_age, y_train_age, X_test_age, y_test_age, 'age', 'age')
    train_model(X_train_race, y_train_race, X_test_race, y_test_race, 'race', 'race')
    
    print(f"\n[COMPLETE] All models have been trained and saved to the '{MODEL_DIR}' directory.")