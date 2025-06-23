# Real-Time Face Feature Detection

This project uses a webcam to perform real-time facial attribute analysis. It detects faces in a live video feed and predicts their age, gender, race, and emotion.

### Features

* **Live Detection:** Utilizes a standard webcam to locate faces in each video frame.
* **Multi-Attribute Analysis:** For each detected face, the system provides the following predictions:
    * **Age:** An estimation of the person's age.
    * **Gender:** Male or Female.
    * **Race:** White, Black, Asian, Indian, or Other.
    * **Emotion:** One of seven classifications (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise).

### Methodology

This project is built using several standard libraries for computer vision and deep learning.

* **OpenCV:** Used for video capture from the webcam and for the initial detection of face locations within the video stream.
* **TensorFlow & Keras:** The core deep learning framework used to build, train, and execute the neural network models.
* **VGG19 (Transfer Learning):** The project uses the VGG19 model, pre-trained on the ImageNet dataset, as a feature extraction base. This base is then fine-tuned on specific datasets to learn the tasks of age, gender, race, and emotion classification.

### Setup and Usage

Follow these steps to set up and run the project.

**1. Prerequisites**
* Python 3.8 or newer.
* A functional webcam connected to your system.

**2. Installation**
* Download or clone the project files (`train_models.py`, `real_time_detection.py`) to a local directory.

**3. Dependencies**
* Navigate to the project directory in your terminal or command prompt and execute the following command to install the required Python libraries:
    ```bash
    pip install opencv-python tensorflow numpy scikit-learn
    ```

**4. Dataset Acquisition**
* The models require specific datasets for training.
    * Create a folder named `datasets` in the root of your project directory.
    * **For Emotion (FER-2013):** Download the dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013). After extracting, ensure the path `datasets/fer2013/` contains the `train` and `test` sub-folders.
    * **For Age, Gender, Race (UTKFace):** Download the `UTKFace.tar.gz` archive from the [project homepage](https://susanqq.github.io/UTKFace/). Extract the archive and place the resulting `UTKFace` folder directly inside your `datasets` folder.

    The final directory structure must be as follows:
    ```
    your_project_folder/
    ├── datasets/
    │   ├── fer2013/
    │   └── UTKFace/
    ├── train_models.py
    ├── train_emotions.py
    └── real_time_detection.py
    ```

**5. Model Training**
* Execute the training script from your terminal. This process is computationally intensive and will take a significant amount of time.
    ```bash
    python train_models.py
    python train_emotions.py
    ```
* Upon completion, new directories named `models_trained_vgg19_full` and `models_trained_vgg19` will be created, containing the trained model `.h5` files.

**6. Execution**
* Once the models have been trained, run the real-time detection script:
    ```bash
    python real_time_detection.py
    ```
* A window will open displaying the webcam feed with detection results overlaid. Press the **'q'** key to terminate the program.
