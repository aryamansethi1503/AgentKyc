import streamlit as st
import google.generativeai as genai
from datetime import date, datetime
from dotenv import load_dotenv
import os
import pytesseract
from PIL import Image
import re
import cv2
import numpy as np
import json

load_dotenv()

st.set_page_config(
    page_title="KYC Verification Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SYSTEM_PROMPT = """
You are a professional KYC verification AI. Your goal is to analyze and compare user-provided form data with text extracted from an identity document.
- Carefully examine the extracted text from the document.
- Compare it against the user's form submission.
- Identify any mismatches based on the strict rules provided for each field.
- Return your findings in a structured JSON format.
"""

def preprocess_image_simple(image: Image.Image) -> Image.Image:
    """Converts to grayscale and applies a basic threshold."""
    img_array = np.array(image.convert('L'))
    _, processed_image = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return Image.fromarray(processed_image)

def extract_text_from_file(file):
    """Extracts text from an uploaded image using Tesseract OCR."""
    try:
        if file.type in ['image/jpeg', 'image/png']:
            image = Image.open(file).convert("RGB")
            preprocessed_image = preprocess_image_simple(image)
            tesseract_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(preprocessed_image, config=tesseract_config)
            return text
        return ""
    except pytesseract.TesseractNotFoundError:
        st.error("Tesseract Error: The Tesseract executable was not found. Please ensure it's installed and in your system's PATH.")
        return ""
    except Exception as e:
        st.error(f"Error during text extraction: {e}")
        return ""

def verify_with_gemini(form_data, ocr_text):
    """
    Sends form data and OCR text to Gemini for verification and returns a structured response.
    """
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        system_instruction=SYSTEM_PROMPT
    )

    prompt = f"""
    Please verify the user's KYC details with extreme precision, following these rules.

    **User's Form Input:**
    - Full Name: {form_data['fullName']}
    - Date of Birth: {form_data['dob'].strftime('%d/%m/%Y')}
    - PAN Number: {form_data['panNumber']}

    **Raw Text Extracted from Document:**
    ```
    {ocr_text}
    ```

    **Your Task & Verification Rules:**
    1.  Analyze the raw text to find the name, date of birth, and PAN number.
    2.  **PAN Number Rule:** The PAN Number must be an **EXACT MATCH**. No flexibility is allowed.
    3.  **Date of Birth Rule:** The Date of Birth is a match if the **year, month, and day components are identical**, regardless of the format or separator (e.g., '17/06/1995' is a match for '1995-06-17').
    4.  **Name Rule:** The name comparison can be flexible (e.g., ignore case, middle initials).
    5.  Return a single JSON object with the verification results as shown in the examples.

    **Example of a successful match with different date formats:**
    {{
        "is_verified": true,
        "mismatches": [],
        "extracted_data": {{"name": "JOHN DOE", "dob": "1995-17-06", "pan": "ABCDE1234F"}},
        "agent_response": "Thank you! All your details have been successfully verified."
    }}
    
    **Example for a failed verification:**
    {{
        "is_verified": false,
        "mismatches": ["PAN Mismatch: You entered 'ABCDE1234F', but the document shows 'ABCDE1234G'."],
        "extracted_data": {{"name": "USER NAME", "dob": "01/01/2000", "pan": "ABCDE1234G"}},
        "agent_response": "I couldn't verify your details. It looks like the PAN number you entered doesn't exactly match the one on the document. Please double-check and try again."
    }}

    Now, process the provided data and return the JSON according to these rules.
    """
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip().lstrip("```json").rstrip("```")
        return json.loads(json_text)
    except json.JSONDecodeError:
        st.error("AI Error: The model returned an invalid format. Please try again.")
        return None
    except Exception as e:
        st.error(f"An error occurred with the AI model: {e}")
        return None

def init_session_state():
    """Initializes all the necessary variables in the session state."""
    if 'api_key_configured' not in st.session_state:
        st.session_state.api_key_configured = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "form_data" not in st.session_state:
        st.session_state.form_data = {
            "fullName": "", "dob": date(2000, 1, 1),
            "panNumber": "", "idCard": None
        }
    if "greeted" not in st.session_state:
        st.session_state.greeted = False

init_session_state()

api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    st.error("Gemini API Key is not set. Please add it to your .env file.")
else:
    try:
        genai.configure(api_key=api_key)
        st.session_state.api_key_configured = True
        if not st.session_state.greeted:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Hello! I'm here to help you with the KYC process. Please fill in your details and upload your PAN card."
            })
            st.session_state.greeted = True
    except Exception as e:
        st.error(f"Invalid API Key or configuration error: {e}")
        st.session_state.api_key_configured = False

if st.session_state.api_key_configured:
    st.title("KYC Verification Agent")
    
    form_col, chat_col = st.columns([1, 1], gap="large")

    with form_col:
        st.header("Your Information")
        with st.form("kyc_form"):
            st.session_state.form_data['fullName'] = st.text_input("Full Name (as on PAN card)", value=st.session_state.form_data['fullName'])
            st.session_state.form_data['dob'] = st.date_input("Date of Birth", min_value=date(1920, 1, 1), value=st.session_state.form_data['dob'])
            st.session_state.form_data['panNumber'] = st.text_input("PAN Number", value=st.session_state.form_data['panNumber'])
            
            st.markdown("---")
            st.subheader("Document Upload")
            st.session_state.form_data['idCard'] = st.file_uploader(
                "Upload PAN Card", type=['jpg', 'jpeg', 'png']
            )
            
            submitted = st.form_submit_button("Submit for Verification", use_container_width=True, type="primary")

            if submitted:
                form_data = st.session_state.form_data
                if not all([form_data['fullName'], form_data['dob'], form_data['panNumber'], form_data['idCard']]):
                    st.warning("Please fill out all fields and upload your PAN card.")
                else:
                    with st.spinner("Analyzing document with AI... This may take a moment."):
                        id_card_text = extract_text_from_file(form_data['idCard'])
                        
                        if not id_card_text.strip():
                             st.error("Could not read the document. Please upload a clearer, high-resolution image and try again.")
                        else:
                            st.write("### Raw OCR Output (for debugging):")
                            st.text_area("Raw Text", id_card_text, height=150)

                            verification_result = verify_with_gemini(form_data, id_card_text)

                            if verification_result:
                                st.write("### AI Analysis Result:")
                                st.json(verification_result.get("extracted_data", "No data extracted."))
                                agent_response = verification_result.get("agent_response", "I seem to be having trouble processing the result. Please try again.")
                                st.session_state.messages.append({"role": "assistant", "content": agent_response})
                                st.rerun()

    with chat_col:
        st.header("💬 KYC Assistant")
        chat_container = st.container(height=500, border=True)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(name=message["role"]):
                    st.markdown(message["content"])

        if user_prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_prompt)
            
            with st.spinner("Thinking..."):
                model = genai.GenerativeModel(model_name='gemini-1.5-flash', system_instruction=SYSTEM_PROMPT)
                response = model.generate_content(user_prompt)
                agent_response = response.text
                
                st.session_state.messages.append({"role": "assistant", "content": agent_response})
                st.rerun()