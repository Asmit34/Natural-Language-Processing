import re
from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini Pro model and start a chat
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Function to get response from Gemini model
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Email validation using regular expression
def is_valid_email(email):
    pattern = r'^\S+@\S+\.\S+$'
    return re.match(pattern, email) is not None

# Phone number validation using regular expression
def is_valid_phone(phone):
    pattern = r'^\d{10}$'
    return re.match(pattern, phone) is not None

# Initialize Streamlit app
st.set_page_config(page_title="Q&A Demo")
st.header("Gemini LLM Application")

# Initialize session state for chat history and form reset if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Text input and button for asking a question
user_question = st.text_input("Input: ", key="user_question")
submit_button = st.button("Ask the question")

# Form for collecting user information
with st.form("user_info_form"):
    st.subheader("Provide your information for a call:")
    user_name = st.text_input("Your Name:")
    user_phone = st.text_input("Phone Number:")
    user_email = st.text_input("Email:")

    # Email validation
    if user_email and not is_valid_email(user_email):
        st.warning("Please enter a valid email address.")
        # Phone number validation
    if user_phone and not is_valid_phone(user_phone):
        st.warning("Please enter a valid phone number.")

    call_request = st.checkbox("I would like a call")

    submit_info_button = st.form_submit_button("Submit")

# Process user input and display response
if submit_button and user_question:
    if user_question.strip().lower() == "i would like a call":
        # Acknowledge the request for a call and prompt for information
        st.session_state['chat_history'].append(("You", user_question))
        st.session_state['chat_history'].append(("Bot", "Sure, please provide your information."))

    else:
        # Continue with the regular question-response flow
        response = get_gemini_response(user_question)
        st.session_state['chat_history'].append(("You", user_question))
        st.subheader("The Response is:")

        # Initialize an empty string to store the aggregated response
        aggregated_response = ""

        # Process each chunk
        for chunk in response:
            st.write(chunk.text)
            aggregated_response += chunk.text

        # Append the aggregated response to the chat history
        st.session_state['chat_history'].append(("Bot", aggregated_response))

# Process user input and display response for form submission
elif submit_info_button:
    # Acknowledge the submission and provide additional information
    st.session_state['chat_history'].append(("Bot", "Great! Please check the information you provided below."))

    # Append valid user information to chat history with HTML line breaks
    if is_valid_email(user_email) and is_valid_phone(user_phone):
        user_info_text = f"Name: {user_name}Phone: {user_phone}Email: {user_email}"
        st.session_state['chat_history'].append(("User Info", user_info_text))
        st.session_state['chat_history'].append(("Bot", "Thank you! I got your information."))

# Display chat history
st.subheader("The Chat History is:")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
