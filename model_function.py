import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        required_columns = ["hours_studied", "score"]
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns: hours_studied and score")

        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def train_model(df):

    X = df[["hours_studied"]]
    y = df["score"]
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    # Save model
    joblib.dump(model, 'model.pkl')  
    return model, r2

def predict_score(hours_studied):
    """Predicting test score based on study hours using saved model."""
    try:
        model = joblib.load('model.pkl')
        prediction = model.predict([[hours_studied]])[0]
        # Ensure score is between 0 and 100
        return np.clip(prediction, 0, 100)  
    except FileNotFoundError:
        raise Exception("Model not found. Please train the model first.")
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")