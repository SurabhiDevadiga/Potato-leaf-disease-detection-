# 🥔 Potato Leaf Disease Detection using Deep Learning

This project is a deep learning-based web application that detects diseases in potato plant leaves using image classification. The system helps farmers and agricultural users identify potato leaf diseases early and take preventive actions.

## 📌 Project Overview

Potato crops are affected by several diseases that reduce productivity and quality. Early detection is important for preventing crop damage. This project uses a Convolutional Neural Network (CNN) model to classify potato leaf images into different disease categories.

The application predicts:

- Potato Early Blight
- Potato Late Blight
- Healthy Potato Leaves

## 🚀 Features

- Upload potato leaf images
- Detect disease using trained deep learning model
- Real-time prediction
- User-friendly interface using Streamlit
- Fast and accurate disease classification

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Streamlit

## 📂 Project Structure

```bash
PotatoDiseaseDetection/
│
├── dataset/
├── model/
├── training/
├── app.py
├── requirements.txt
├── saved_model/
├── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/SurabhiDevadiga/Potato-leaf-disease-detection-.git
```

Move into project directory:

```bash
cd Potato-leaf-disease-detection-
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 📊 Model Workflow

1. Collect potato leaf dataset  
2. Preprocess images  
3. Train CNN model  
4. Save trained model  
5. Deploy using Streamlit  
6. Predict disease from uploaded images  

## 🎯 Future Enhancements

- Support multiple crop diseases
- Improve model accuracy
- Deploy using cloud platforms
- Mobile application integration

## 👩‍💻 Author

Surabhi Devadiga

GitHub: https://github.com/SurabhiDevadiga
