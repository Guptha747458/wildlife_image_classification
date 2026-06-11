# 🐾 Wildlife Image Classification with KNN and Naive Bayes

A Streamlit-based web application that classifies animal images using K-Nearest Neighbors (KNN) and Naive Bayes models trained on image histograms.

## 🚀 Features

- **User-Friendly Interface**: Upload images (JPG, JPEG, PNG) through a clean, modern Streamlit UI.
- **Histogram-based Feature Extraction**: Extracts grayscale pixel intensity distribution histograms (normalized and flattened) from uploaded images.
- **Dual Model Prediction**: Classifies images using two classic machine learning models:
  - **K-Nearest Neighbors (KNN)**
  - **Naive Bayes (Gaussian NB)**
- **Real-Time Classification**: Displays the prediction results from both models side by side instantly.

---

## 🛠️ Project Structure

- [wildlife.py](file:///c:/Users/HAI/OneDrive/Documents/wildlife_image_classification/wildlife.py): The main Streamlit web application.
- [requirements.txt](file:///c:/Users/HAI/OneDrive/Documents/wildlife_image_classification/requirements.txt): Python dependencies for the project.
- `knn_model.pkl`: Trained K-Nearest Neighbors model.
- `nb_model.pkl`: Trained Naive Bayes model.
- `label_encoder.pkl`: Fitted scikit-learn LabelEncoder mapping class indices to animal names.

---

## 💻 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/wildlife_image_classification.git
cd wildlife_image_classification
```

### 2. Install the requirements
Install the required packages. Note that you will need `streamlit` and `joblib` as well:
```bash
pip install -r requirements.txt
pip install streamlit joblib
```

### 3. Running the Application
Launch the Streamlit app using the following command:
```bash
streamlit run wildlife.py
```
This will start a local server and open the app automatically in your default browser (usually at `http://localhost:8501`).

---

## 📷 How It Works

1. **Upload**: Drag and drop or upload an animal image (e.g., cat, dog, lion, elephant).
2. **Feature Extraction**: The app converts the image to grayscale, resizes it to $64 \times 64$ pixels, and calculates a normalized 256-bin pixel intensity histogram.
3. **Classification**: 
   - The histogram feature vector is passed to the pre-trained models.
   - Predictions from both the **KNN** model and the **Naive Bayes** model are displayed.

---

## 🌐 Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended & Free)
Streamlit provides a free, zero-config hosting platform directly integrated with GitHub:

1. **Push your code to GitHub**: Create a repository and push all your project files (`wildlife.py`, `requirements.txt`, `.pkl` model files).
2. **Sign up**: Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. **Deploy App**:
   - Click **New app**.
   - Select your repository, branch (usually `main`), and the main file path (`wildlife.py`).
   - Click **Deploy!**
4. Your application will be live at a URL like `https://<your-app-name>.streamlit.app/`.

### Option 2: Hugging Face Spaces (Free & Simple)
Hugging Face Spaces supports Streamlit out of the box:

1. Create a free account on [Hugging Face](https://huggingface.co/).
2. Click on **Spaces** and choose **Create new Space**.
3. Fill in the Space name, select **Streamlit** as the SDK, and choose a free hardware tier.
4. Clone the space locally or upload your files (`wildlife.py`, `requirements.txt`, `.pkl` files) directly via the Hugging Face web interface.
5. Hugging Face will build and serve your app automatically!

