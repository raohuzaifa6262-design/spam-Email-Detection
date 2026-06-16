import streamlit as st
import pickle
import sklearn

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# App title
st.title("📧 Spam Detection App (SVM)")
st.write("Enter your message/email below to check if it's Spam or Not")

# Input box
user_input = st.text_area("✍️ Enter Message")

# Button
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        # Transform input
        input_vect = vectorizer.transform([user_input])
        
        # Prediction
        result = model.predict(input_vect)[0]
        
        # Output
        if result == 1:
            st.error("🚫 This is SPAM message")
        else:
            st.success("✅ This is NOT SPAM")

# Footer
st.markdown("---")
st.caption("Developed using SVM & Streamlit")