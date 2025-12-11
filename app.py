import joblib
import re
import streamlit as st
import pandas as pd
import time
import nltk
from nltk.tokenize import sent_tokenize
from nltk.data import find

# emoji definitions
emoji_check   = "✅"
emoji_stop    = "⛔️"
emoji_list    = "📋"
emoji_plus    = "➕"
emoji_garbage = "🗑️"


def ensure_nltk(nm):
    try:
        find(nm)
    except LookupError:
        nltk.download(nm)

ensure_nltk('punkt')
ensure_nltk('punkt_tab')

st.set_page_config(page_title="SpamSentry",page_icon="📨")
style = '''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,100..900;1,100..900&display=swap');
        html, body, [class*="css"], h1, h2, h3, h4, h5 {
            font-family: "Montserrat", sans-serif !important;
            font-optical-sizing: auto;
            font-weight: 400;
            font-style: normal;
        }
        section[data-testid="stSidebar"] div.stElementContainer {
            width: 100%;
            text-align: left;
            justify-content: left;
        }
    </style>
'''

st.markdown(style, unsafe_allow_html = True)

# Load the model and vectorizer
model = joblib.load('spam_model.pkl')

# Function to predict if a message is spam
def predict_user_input(user_input):
    for user_word in re.split('-| |_', user_input):
        if len(user_word) > 25:
            return True # spam
    processed_input = pd.DataFrame({
        "message": [user_input],
        "char_count": [len(user_input)],
        "word_count": [len(user_input.split())],
        "sentence_count": [user_input.count('.') + user_input.count('!') + user_input.count('?')]
    })
    prediction = model.predict(processed_input)
    return prediction[0]  # returns False for not spam, True for spam

# Set up the Streamlit application
st.sidebar.image("logo.png")
st.sidebar.subheader("Spam Detection Algorithm")

# Sidebar for navigation
options = ["Single Message Check", "Multiple Messages Check", "Info"]
logos = ["📩", "📨", "📋"]

# Initializing state variables
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'choice' not in st.session_state:
    st.session_state['choice'] = options[2]

for i, c in enumerate(options):
    if st.sidebar.button(f"{logos[i]} {c}"):
        st.session_state['choice'] = c
st.sidebar.divider()
st.sidebar.link_button("🖥️ GitHub Repo (🔗)","https://github.com/Vardan2009/spamsentry")
st.sidebar.link_button("📋 Examples (🔗)","https://github.com/Vardan2009/spamsentry/blob/main/EXAMPLES.md")

if st.session_state['choice'] == "Single Message Check":
    st.header(f"{logos[0]} Single Message Check")
    user_input = st.text_area("Enter your message")
    
    if st.button(f"{emoji_check} Check"):
        if not user_input:
            st.warning(f"{emoji_stop} Please enter a message to check.")
        else:
            with st.spinner('Please Wait...'):
                time.sleep(1)
                result = predict_user_input(user_input)
                if not result:
                    st.success(f"{emoji_check} The message is safe")
                else:
                    st.error(f"{emoji_stop} The message is spam")
elif st.session_state['choice'] == "Multiple Messages Check":
    st.header(f"{logos[1]} Multiple Messages Check")
    message = st.text_area("Enter your message")
    col1, _, col2 = st.columns([1,7,1])
    with col2:
        if st.button(f"{emoji_plus} Add") and len(message) > 0:
            st.session_state['messages'].append(message)
    with col1:
        check = st.button(f"{emoji_check} Check")
    if check:
        if len(st.session_state["messages"]) <= 0:
            st.warning(f"{emoji_stop} Please enter messages to check")
        else:
            processed_input = pd.DataFrame({
                "message": st.session_state['messages'],
                "char_count": [len(user_input) for user_input in st.session_state['messages']],
                "word_count": [len(user_input.split()) for user_input in st.session_state['messages']],
                "sentence_count": [len(sent_tokenize(user_input)) for user_input in st.session_state['messages']]
            })
            predictions = model.predict(processed_input)
            for i, message in enumerate(st.session_state['messages']):
                if predictions[i]:
                    st.error(f'{emoji_stop} {message}')
                else:
                    st.success(f'{emoji_check} {message}')
            del st.session_state['messages']
    else:
        # Process button clicks and remove messages
        for i in range(len(st.session_state['messages'])):
            message = st.session_state['messages'][i]
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            with c1:
                st.info(message)
            with c2:
                if st.button(emoji_garbage, key=f"delete_{i}"):
                    st.session_state['messages'].remove(message)
                    st.rerun()
                    break
elif st.session_state['choice'] == "Info":
    st.header(f"{logos[2]} About This Application")
    st.write("""
    This application uses a machine learning model to detect spam messages.
    It provides two main functionalities:
    
    1. **Single Message Check**: Enter a message to check if it's spam.
    2. **Multiple Messages Check**: Upload multiple messages to check each one for spam.

    The model was trained on a dataset of spam and not spam messages. The output is a simple indication of whether the message is considered spam or not.

    #### How to Use:
    - For a single message, type it in the text area and click 'Check'.
    - For multiple messages, type multiple messages and click 'Check'.

    #### Credits:
    Created as a final work for the "Data is King" workshop in TUMO.
    - Narek Shaghoyan
    - Yeva Mandaryan
    - Vardan Petrosyan
    - Ashot Meliqyan
    - Anush Grigoryan
    """)
