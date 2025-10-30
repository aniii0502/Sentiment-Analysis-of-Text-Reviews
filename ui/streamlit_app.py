import streamlit as st
import requests

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🤖",
    layout="wide"
)

st.title("Sentiment Analysis App 🤖")
st.markdown("Enter a movie or product review below, and the AI will classify its sentiment as positive or negative.")

API_URL = "http://127.0.0.1:5000/predict"

with st.form(key='review_form'):
    user_input = st.text_area(
        label="Enter your review:", 
        height=150,
        placeholder="e.g., 'This movie was fantastic!' or 'I was very disappointed with this product.'"
    )
    
    analyze_button = st.form_submit_button(label="Analyze Sentiment")

if analyze_button and user_input:
    with st.spinner("Analyzing..."):
        try:
            payload = {"text": user_input}
            
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                prediction = response.json()
                sentiment = prediction.get('sentiment')
                
                if sentiment == 'positive':
                    st.success(f"**Sentiment: Positive** 😊")
                else:
                    st.error(f"**Sentiment: Negative** 😞")
                
                with st.expander("Show API Response"):
                    st.json(prediction)
                    
            else:
                st.error(f"Error from API ({response.status_code}): {response.json().get('error', 'Unknown error')}")

        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Could not connect to the API. Is the Flask server (api/app.py) running?")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

elif analyze_button and not user_input:
    st.warning("Please enter some text to analyze.")

st.sidebar.header("About")
st.sidebar.info(
    "This is a full-stack web application that uses a machine learning model "
    "to perform sentiment analysis on text reviews. \n\n"
    "The stack includes:\n"
    "- **Python**\n"
    "- **Scikit-learn** (for the model)\n"
    "- **Flask** (for the backend API)\n"
    "- **Streamlit** (for this UI)"
)
