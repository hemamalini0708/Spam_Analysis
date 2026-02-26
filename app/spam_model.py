# Import streamlit library - this creates the web interface automatically
import streamlit as st
# Import pickle to load the trained model
import pickle
# Import string to remove punctuation
import string
# Import nltk for natural language processing
import nltk
# Import stopwords (common words like 'the', 'is' that don't add meaning)
from nltk.corpus import stopwords
# Import WordNetLemmatizer to convert words to their root form
from nltk.stem import WordNetLemmatizer
# Import one_hot for text encoding (converts words to numbers)
from tensorflow.keras.preprocessing.text import one_hot
# Import pad_sequences to make all texts same length for model
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Import numpy for numerical operations
import numpy as np

# ============ DOWNLOAD REQUIRED NLTK DATA ============
# Download tokenizer data (splits text into words)
nltk.download('punkt')
# Download stopwords data (common English words to remove)
nltk.download('stopwords')
# Download wordnet data (for lemmatization - converting to root words)
nltk.download('wordnet')

# ============ PAGE CONFIGURATION ============
# Set the page title and layout
st.set_page_config(
    page_title="Spam Detector",  # Browser tab title
    layout="centered"  # Center the content on page
)

# ============ LOAD MODEL ============
# Open the saved model file in read-binary mode
with open('spam_review.pkl', 'rb') as f:
    # Load the pickled model into memory
    model = pickle.load(f)

# ============ INITIALIZE TEXT PROCESSOR ============
# Create a lemmatizer object (converts words to root form)
# Example: running, runs, ran → run
lemma = WordNetLemmatizer()

# ============ MAIN STREAMLIT CODE ============

# Display main heading on the page
st.title("📧 Spam Detection System")

# Display a subheading with explanation
st.subheader("Check if your message is Spam or Ham")

# Display informational text
st.info("This AI model uses Deep Learning to detect spam messages with high accuracy!")

# ============ TEXT INPUT SECTION ============
# Create a text area where users can input their message
# The input is stored in the variable 'user_input'
user_input = st.text_area(
    label="Enter your message here:",  # Label shown above the text box
    height=150,  # Height of the text box in pixels
    placeholder="Paste your message to check..."  # Gray placeholder text
)

# ============ PREDICTION SECTION ============
# Create a button for user to click for prediction
if st.button("Analysis Message", use_container_width=True):

    # Check if user has entered text
    if user_input:

        # Display a spinner animation while processing
        with st.spinner("Analyzing message..."):

            # ============ TEXT PREPROCESSING ============
            # Convert all text to lowercase (so 'SPAM' and 'spam' are treated same)
            text = user_input.lower()

            # Remove all punctuation marks (!@#$%^&*. etc)
            # Keep only letters, numbers and spaces
            text = ''.join([i for i in text if i not in string.punctuation])

            # Remove stopwords (common words that don't add meaning)
            # Also lemmatize each word (convert to root form)
            # This line:
            # 1. Splits text into words: text.split()
            # 2. Removes stopwords: if i not in stopwords.words('english')
            # 3. Converts to root form: lemma.lemmatize(i)
            # 4. Joins back: ' '.join(...)
            text = ' '.join([lemma.lemmatize(i) for i in text.split() if i not in stopwords.words('english')])

            # ============ MODEL PREPROCESSING ============
            # Convert text to numbers using one_hot encoding
            # 5500 = vocabulary size (number of unique words model knows)
            # Returns a list with one encoded text
            v = [one_hot(text, 5500)]

            # Pad all sequences to same length (953 words)
            # If text is shorter: add zeros at the end (padding='post')
            # If text is longer: cut it to 953 words
            # All texts must be same length for the model
            p = pad_sequences(v, maxlen=953, padding='post')

            # ============ MAKE PREDICTION ============
            # Feed the preprocessed text to the model
            # Model returns a probability between 0 and 1
            prediction = model.predict(p)

            # Extract the probability value (it's in a nested array)
            confidence = float(prediction[0][0])

            # ============ INTERPRET RESULT ============
            # If probability > 0.5, it's Ham (legitimate)
            # If probability <= 0.5, it's Spam
            if confidence > 0.5:
                result = "HAM (Legitimate)"
                color = "green"
            else:
                result = "SPAM (Suspicious)"
                color = "red"

            # ============ DISPLAY RESULTS ============
            # Add some spacing
            st.markdown("---")

            # Display result in colored box
            st.markdown(f"### Result: <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)

            # Display confidence percentage
            confidence_percent = confidence * 100
            st.metric(
                label="Confidence Score",  # Metric label
                value=f"{confidence_percent:.2f}%"  # Show 2 decimal places
            )

            # Display a progress bar showing confidence
            st.progress(confidence)

            # Display the preprocessed text (so user can see what model saw)
            with st.expander("View Processed Text"):
                st.write(text)

    # If user clicks button without entering text
    else:
        # Display warning message
        st.warning("Please enter a message first!")

# ============ FOOTER SECTION ============
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Project Done By Hema Malini Powered by Deep Learning | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)