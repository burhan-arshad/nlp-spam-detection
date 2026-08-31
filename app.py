import streamlit as st
import joblib

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("spam_classifier.pkl")

model = load_model()

st.title("SMS Spam Classifier")

st.write(
    "Enter an SMS message below and the machine learning model "
    "will predict whether it is **Spam** or **Ham**."
)

st.divider()

message = st.text_area(
    "Enter your message",
    placeholder="Example: Congratulations! You have won a free prize. Click here to claim it.",
    height=150
)

if st.button("Classify Message", use_container_width=True):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        prediction = model.predict([message])[0]

        if prediction == 1:
            st.error("SPAM")
            st.write("This message is likely to be spam.")

        else:
            st.success("HAM")
            st.write("This message appears to be legitimate.")

st.divider()

st.subheader("Try an Example")

col1, col2 = st.columns(2)

with col1:
    if st.button("Spam Example", use_container_width=True):
        st.info(
            "Congratulations! You have won a £1000 prize. "
            "Call now to claim your reward."
        )

with col2:
    if st.button("Ham Example", use_container_width=True):
        st.info(
            "Hey, are we still meeting for dinner tonight?"
        )

st.divider()

st.subheader("About the Model")

st.write(
    "This application uses a Linear Support Vector Machine (LinearSVC) "
    "combined with TF-IDF text representation."
)

st.write(
    "The final model was optimized using GridSearchCV."
)

st.caption("Developed by Burhan Arshad")