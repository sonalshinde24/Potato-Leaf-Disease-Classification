🥔 Potato Leaf Disease Classification using CNN (Streamlit App)

An end-to-end Deep Learning web application to classify potato leaf diseases using a Convolutional Neural Network (CNN) model built with Keras and deployed using Streamlit.

The app allows users to upload a potato leaf image and instantly predicts whether the leaf is healthy or affected by a disease.

🔗 Live App:
👉 https://potato-leaf-disease-classification-gjpjedyhjyruqulnnkhpbe.streamlit.app/

📊 Dataset:
👉 https://www.kaggle.com/datasets/arjuntejaswi/plant-village

-----

📌 Project Overview

Potato crops are highly susceptible to leaf diseases that can significantly reduce yield if not detected early.
This project uses Deep Learning (CNN) to automatically identify potato leaf conditions from images, helping in early disease detection.

The model is trained on the PlantVillage dataset and deployed as an interactive Streamlit web app.

-----

🧠 Diseases Classified

The model classifies potato leaf images into the following categories:

🟢 Healthy
🟤 Early Blight
⚫ Late Blight

-----

🛠️ Tech Stack

- Programming Language: Python
- Deep Learning Framework: TensorFlow / Keras
- Model Type: Convolutional Neural Network (CNN)
- Web Framework: Streamlit
- Image Processing: OpenCV, PIL
- Deployment: Streamlit Cloud

📂 Project Structure
```bash
.
├── app.py                     # Streamlit application
├── potato_disease_model.keras # Trained CNN model
├── background.jpg             # Background image for UI
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```
---

🧪 Model Details

- Architecture: Custom CNN
- Input Size: 224 × 224 × 3
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Evaluation Metric: Accuracy
- Model Format: .keras
- The trained model is saved and directly loaded into the Streamlit app for inference.

🚀 How the App Works

- User uploads a potato leaf image.
- Image is resized and preprocessed.
- CNN model predicts the disease class.
- Prediction result is displayed instantly on the UI.

▶️ Run Locally
1️⃣ Clone the Repository
```bash
git clone https://github.com/sonalshinde24/potato-leaf-disease-classification.git
cd potato-leaf-disease-classification
```
2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

🎯 Results

- High accuracy in classifying potato leaf diseases
- Fast and user-friendly web interface
- Real-time disease prediction from images

🌱 Future Improvements

- Add confidence score for predictions
- Support for more crop diseases
- Mobile-friendly UI
- Model optimization for faster inference
- Disease treatment recommendations

🙌 Acknowledgements

- PlantVillage Dataset for providing high-quality agricultural images
- Kaggle community
- Streamlit for easy deployment
