import streamlit as st
from model_function import train_model, predict_score, load_data

def main():
    # Sidebar with Instructions and Information
    st.sidebar.title("📌 Important Information")
    st.sidebar.markdown(
        """
        ### 📝 Key Information:
        - The model was trained using **~2480 data points** of study hours and corresponding scores.
        - **Score Prediction Range**:  
          - For study hours between **0 to 11**, the model predicts your score accurately.  
          - For inputs above **11 hours**, the model predicts **100%**, reflecting real-life limits where over-preparation can reduce effectiveness.
        
        
        ### 💡 Life Tip: The 8-8-8 Rule
        To achieve balance and long-term success, follow this golden rule:  
        - 📖 **8 Hours Study**: Focused and distraction-free learning.  
        - 😴 **8 Hours Sleep**: To recharge your mind and body.  
        - 🧘‍♀️ **8 Hours Rest & Leisure**: For hobbies, relaxation, and spending time with loved ones.  
        
        
        ### 🌟 Quick Reminder:
        - Study smart, not just hard—quality over quantity matters.
        - Your well-being is as important as your success. Take care of yourself!  
        
        
        **Thank you for spending your valuable time here!** 😊
        """
    )

    # Main Title and Subtitle
    st.title("📚 Study Hours Score Predictor")
    st.write("Predict your test score based on the number of hours you study each day.")

    # Load Dataset
    sample_data_path = "student_scores.csv"
    df = None

    try:
        df = load_data(sample_data_path)
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")

    if df is not None:
        # Train the Model
        st.subheader("📊 Model Training")
        model, r2 = train_model(df)
        
        # Showing R² value in the app
        st.write(f" ### **R² Score**: {r2:.4f}")
        st.write(f"#### 96.32% means that this model is very good at predicting the scores based on study hours. It explains 96.32% of the data accurately, which means the predictions are very close to the actual results. Only about 3.68% of the variation in the data is not explained by the model. 👍")

        st.markdown( """
        #### Definition of R² : 
        ##### R² (R-squared) is a statistical measure that indicates how well the regression model explains the variability of the dependent variable.""")
        
        # Prediction Section
        st.subheader("🔮 Predict Your Score")
        hours = st.number_input(
            "Enter study hours (0-11):",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.1
        )

        if st.button("Predict"):
            try:
                predicted_score = predict_score(hours)
                st.success(f"🎯 Predicted Score: {predicted_score:.2f}%")
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

    # Footer
    st.markdown(
        """
        ---
        Thank you for using the Study Hours Score Predictor! 😊
        """
    )

if __name__ == "__main__":
    main()