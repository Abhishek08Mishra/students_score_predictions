# 📚 Study Hours Score Predictor

A machine learning project to predict student test scores based on the number of study hours using Linear Regression. The project demonstrates how to use supervised learning for predicting continuous outcomes.

---

## ⚙️ Project Overview

This project uses a **Linear Regression** model to predict students' test scores based on their study hours. The model was trained using **~2,500 data points** and achieves a high **R² score of 96.32%**, indicating that it can predict scores with great accuracy.

---

## 🛠️ Key Features

- **Train a Model**: The model is trained using historical data of study hours and scores.
- **Predict Scores**: Input your study hours, and the model will predict your test score.
- **Beautiful UI**: Built using **Streamlit**, providing an easy-to-use interface for predictions.
- **Real-World Data**: The dataset contains **2,500 data points** with hours studied and their corresponding scores.

---

## 💡 Key Information

- **R² Score**: The model explains **96.32%** of the variation in test scores based on study hours, making the predictions highly accurate.
- **Prediction Range**: The model predicts scores for study hours between **0 and 11 hours**. For inputs above 11 hours, the model predicts **100%** to reflect real-world scenarios where over-preparation leads to diminishing returns.
- **8-8-8 Rule**:  
   - 📖 **8 Hours Study**  
   - 😴 **8 Hours Sleep**  
   - 🧘‍♀️ **8 Hours Rest and Leisure**  
  Follow this rule to live a balanced and productive life.

---

## 📊 How It Works

1. **Data Collection**: The dataset consists of **2,480** student records with study hours and their respective scores.
2. **Model Training**: The data is split into training and testing sets. A **Linear Regression** model is trained on the data.
3. **Prediction**: After training, the model can predict the score for any given number of study hours (within the prediction range).
4. **Evaluation**: The model’s performance is evaluated using the **R² score**, which is **96.32%** for this dataset.

---

## 🌍 Live Demo

You can interact with the model by visiting the live Streamlit app (if hosted). The app allows users to input study hours and predict their score. Stay tuned for the link to the live demo!

---

## 🛠️ Built Using

- Python
- 🎈 Streamlit
- 🐼 Pandas
-  🔢 Numpy
- 🧠 Scikit-learn
- 📖 Joblib


## 🔧 Installation & Usage

To run the project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/study-hours-score-predictor.git
   cd study-hours-score-predictor
2. Install the required dependencies:
        pip install -r requirements.txt

3. Launch!
    streamlit run app.py

## 🤝 Want to Contribute?

Yes please! Here's how:

1. Fork it
2. Create your feature branch (`git checkout -b cool-feature`)
3. Commit your changes (`git commit -am 'Added something cool'`)
4. Push to the branch (`git push origin cool-feature`)
5. Start a Pull Request

## 📬 Get in Touch

Got questions? Reach out!

- 🐦 Twitter: [(https://x.com/Abhi__57)]
- 💼 LinkedIn : [(https://www.linkedin.com/in/abhishek-mishra-120799281/)]
- 📧 Email: abishekmishra195@gmail.com


## ⭐ Show Your Support

If this projects helps you, show some love by giving it a star! ⭐

---
Made with ❤️ by [Abhishek Mishra]

