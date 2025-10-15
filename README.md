# Real-Time Speech Emotion Recognition

This project is a deep learning-based application designed to recognize human emotions from speech audio. The system extracts various audio features, trains a model to classify emotions, and deploys it as a web application using Flask. The repository includes a pre-trained model, `stackclassifier.pkl`, allowing for immediate use.

<img width="500" height="500" alt="Screenshot 2025-10-16 020241" src="https://github.com/user-attachments/assets/4e14935e-aa98-49af-8cb2-f4f1304056cd" />


## 📋 Features
- **Pre-Trained Model**: Includes a pre-trained classifier (`stackclassifier.pkl`) for immediate predictions.
- **Feature Extraction**: Extracts key audio features including MFCC, Zero-Crossing Rate (ZCR), and Root Mean Square (RMS) energy using the Librosa library.
- **Model Architecture**: The provided model is an `MLPClassifier`. The training script also includes code for a custom deep learning model.
- **Web-Based Interface**: Deployed as a simple web application using Flask, allowing users to upload an audio file and get a real-time emotion prediction.

## 💻 Technologies Used
- **Python 3.x**
- **Core Libraries**:
    - **TensorFlow & Keras**: For building and training the deep learning model.
    - **Librosa**: For audio processing and feature extraction.
    - **Scikit-learn**: For machine learning utilities and the MLPClassifier.
    - **Pandas & NumPy**: For data manipulation and numerical operations.
    - **Flask**: For deploying the model as a web application.

## 🚀 Running the Live Demo

This project comes with a pre-trained model, so you can run the web application directly.

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

**2. Create a virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install the required libraries:**
```bash
pip install -r requirements.txt
```

**4. Run the Flask application:**
Execute the Flask app script from your terminal.
```bash
python flask_app.py
```

**5. Use the web interface:**
- Open your web browser and navigate to `http://127.0.0.1:5000`.
- Upload a `.wav` audio file containing speech.
- The application will display the predicted emotion.

---

## 🧠 Training the Model from Scratch (Optional)

If you wish to train the model yourself, you will need the original dataset.

**1. Download the Dataset:**
- Download the RAVDESS dataset from the dataset folder or [this link](https://zenodo.org/record/1188976).
- Unzip the files and place them in a designated `data` folder within the project directory.

**2. Run the Training Script:**
- Execute the `speechrecognition_project.py` script. This will process all the audio files, extract features, train a new model, and save it.
