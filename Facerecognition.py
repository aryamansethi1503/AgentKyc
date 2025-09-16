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
import face_recognition

load_dotenv()
st.set_page_config(page_title="Intelligent KYC Agent", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

try:
    FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.error("Error loading face cascade model. Make sure 'haarcascade_frontalface_default.xml' is in the same folder as the script.")
    st.stop()

VERIFICATION_PROMPT = """
You are a professional KYC verification AI. Your goal is to analyze and compare user-provided form data with text extracted from an identity document.
- Carefully examine the extracted text from the document.
- Compare it against the user's form submission.
- Identify any mismatches based on the strict rules provided for each field.
- Return your findings in a structured JSON format.
- Your response must be polite, clear, and concise.
"""
CHATBOT_PROMPT = """
You are a helpful and friendly KYC Assistant. Your role is to answer user questions about the KYC process.
- Be clear, concise, and professional.
- Do not provide financial advice.
- If you don't know the answer, say that you cannot help with that question.
- DO NOT output your response in JSON format. Respond in plain, conversational text.
"""

def is_text_meaningful(text):
    return len(re.findall(r'[a-zA-Z0-9]', text)) >= 5

def extract_face_from_id(image: Image.Image):
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0: return None
    (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    return Image.fromarray(img_array[y:y+h, x:x+w])

def get_face_encoding(image: Image.Image):
    img_array = np.array(image.convert('RGB'))
    encodings = face_recognition.face_encodings(img_array)
    return encodings[0] if encodings else None

def compare_faces(encoding1, encoding2):
    if encoding1 is None or encoding2 is None: return False, 999.0
    match = face_recognition.compare_faces([encoding1], encoding2, tolerance=0.6)
    distance = face_recognition.face_distance([encoding1], encoding2)[0]
    return match[0], distance

def preprocess_image_simple(image: Image.Image):
    img_array = np.array(image.convert('L'))
    _, processed_image = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return Image.fromarray(processed_image)

def extract_text_from_file(file):
    try:
        image = Image.open(file).convert("RGB")
        preprocessed_image = preprocess_image_simple(image)
        text = pytesseract.image_to_string(preprocessed_image)
        return text, Image.open(file)
    except Exception as e:
        st.error(f"Error during OCR: {e}")
        return "", None

def extract_initial_data_with_gemini(ocr_text):
    model = genai.GenerativeModel(model_name='gemini-1.5-flash')
    prompt = f"""
    You are a data extraction assistant. From the following OCR text from an Indian PAN card, extract the Name, Date of Birth (in DD/MM/YYYY format), and the PAN number.
    OCR TEXT:
    ```{ocr_text}```
    Return a single JSON object with the keys "name", "dob", and "pan". If a value cannot be found, return an empty string for it.
    Example: {{"name": "JOHN DOE", "dob": "15/03/2002", "pan": "ABCDE1234F"}}
    """
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip().lstrip("```json").rstrip("```")
        return json.loads(json_text)
    except Exception:
        return {"name": "", "dob": "", "pan": ""}

def verify_with_gemini(form_data, ocr_text):
    model = genai.GenerativeModel(model_name='gemini-1.5-flash', system_instruction=VERIFICATION_PROMPT)
    prompt = f"""
    Please verify the user's KYC details with extreme precision.
    **User's Form Input:**
    - Full Name: {form_data['fullName']}
    - Date of Birth: {form_data['dob']}
    - PAN Number: {form_data['panNumber']}
    **Raw Text Extracted from Document:**
    ```{ocr_text}```
    **Verification Rules:**
    1.  **PAN Number Rule:** The PAN Number must be an **EXACT MATCH**.
    2.  **Date of Birth Rule:** The year, month, and day components must be identical, regardless of format.
    3.  **Name Rule:** The name comparison can be flexible.
    4.  **CRITICAL DATA PRESENCE RULE:** You MUST find a plausible Name, Date of Birth, AND PAN Number in the 'Raw Text Extracted from Document'. If any of these three pieces of information are missing from the document's text, you MUST fail the verification. Set "is_verified" to false and state in the "agent_response" that the document is invalid because it's missing required information.
    **Your Task:**
    Return a single JSON object with the following specific structure:
    {{
        "is_verified": boolean,
        "mismatches": ["A list of strings describing each mismatch found."],
        "agent_response": "A polite, concise message to the user in plain text explaining the final result (either success or the specific reason for failure)."
    }}
    """
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip().lstrip("```json").rstrip("```")
        return json.loads(json_text)
    except Exception as e:
        st.error(f"AI Error: {e}")
        return None

def init_session_state():
    if "api_key_configured" not in st.session_state: st.session_state.api_key_configured = False
    if "messages" not in st.session_state: st.session_state.messages = []
    if "greeted" not in st.session_state: st.session_state.greeted = False
    if "step" not in st.session_state: st.session_state.step = 1
    if "id_image" not in st.session_state: st.session_state.id_image = None
    if "id_face_encoding" not in st.session_state: st.session_state.id_face_encoding = None
    if "selfie_image" not in st.session_state: st.session_state.selfie_image = None
    if "face_match_result" not in st.session_state: st.session_state.face_match_result = None
    if "ocr_text" not in st.session_state: st.session_state.ocr_text = ""
    if "extracted_form_data" not in st.session_state: st.session_state.extracted_form_data = {}
init_session_state()

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    st.error("Gemini API Key is not set...")
else:
    try:
        genai.configure(api_key=api_key)
        st.session_state.api_key_configured = True
        if not st.session_state.greeted:
            st.session_state.messages.append({"role": "assistant", "content": "Hello! Please upload your document to begin."})
            st.session_state.greeted = True
    except Exception as e:
        st.error(f"Invalid API Key or configuration error: {e}")

if st.session_state.api_key_configured:
    st.title("Intelligent KYC Verification Agent")
    form_col, chat_col = st.columns([1, 1], gap="large")

    with form_col:
        if st.session_state.step == 1:
            st.header("Step 1: Upload Your Document")
            uploaded_file = st.file_uploader("Upload PAN Card", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                with st.spinner("Processing document..."):
                    st.session_state.ocr_text, st.session_state.id_image = extract_text_from_file(uploaded_file)
                    if not is_text_meaningful(st.session_state.ocr_text):
                        st.error("This does not appear to be a valid document. No meaningful text could be read.")
                    else:
                        st.session_state.extracted_form_data = extract_initial_data_with_gemini(st.session_state.ocr_text)
                        id_face_image = extract_face_from_id(st.session_state.id_image)
                        if id_face_image:
                            st.session_state.id_face_encoding = get_face_encoding(id_face_image)
                            if st.session_state.id_face_encoding is not None:
                                st.success("✅ Document text and face detected.")
                                st.session_state.step = 2
                                st.rerun()
                            else: st.error("Could not read face from the ID. Please use a clearer photo.")
                        else: st.error("No face detected on the ID document.")

        if st.session_state.step == 2:
            st.header("Step 2: Biometric Verification")
            st.info("Please look at the camera and take a clear selfie.")
            selfie_img_buffer = st.camera_input("Take Selfie")
            if selfie_img_buffer:
                with st.spinner("Analyzing selfie..."):
                    selfie_pil = Image.open(selfie_img_buffer).convert("RGB")
                    selfie_encoding = get_face_encoding(selfie_pil)
                    if selfie_encoding is not None:
                        is_match, distance = compare_faces(st.session_state.id_face_encoding, selfie_encoding)
                        st.session_state.face_match_result = (is_match, distance)
                        st.session_state.step = 3
                        st.rerun()
                    else: st.error("Could not detect a face in the selfie. Please try again.")

        if st.session_state.step == 3:
            st.header("Step 3: Correct and Confirm Your Details")
            st.info("Our AI has pre-filled the form. Please correct any errors before submitting.")
            
            st.subheader("Raw Extracted Text")
            st.text_area(
                label="Text read from your document by the OCR engine:",
                value=st.session_state.get("ocr_text", "No text was extracted."),
                height=150,
                disabled=True
            )

            is_match, distance = st.session_state.face_match_result
            if is_match:
                st.success(f"✅ Face Match Successful! (Confidence Score: {1-distance:.2%})")
            else:
                st.error(f"❌ Face Match Failed. (Confidence Score: {1-distance:.2%})")
            
            if st.button("⬅️ Retake Selfie"):
                st.session_state.step = 2
                st.rerun()

            with st.form("kyc_form"):
                fullName = st.text_input("Full Name", value=st.session_state.extracted_form_data.get("name", ""))
                dob_str = st.text_input("Date of Birth (DD/MM/YYYY)", value=st.session_state.extracted_form_data.get("dob", ""))
                panNumber = st.text_input("PAN Number", value=st.session_state.extracted_form_data.get("pan", ""))
                submitted = st.form_submit_button("Confirm and Verify", use_container_width=True, type="primary", disabled=not is_match)
                
                if submitted:
                    form_data = {"fullName": fullName, "dob": dob_str, "panNumber": panNumber}
                    with st.spinner("Running final AI verification..."):
                        verification_result = verify_with_gemini(form_data, st.session_state.ocr_text)
                        if verification_result:
                            agent_response = verification_result.get("agent_response", "Verification process is complete.")
                            st.session_state.messages.append({"role": "assistant", "content": agent_response})
                            st.session_state.step = 4
                            st.rerun()
        
        if st.session_state.step == 4:
            st.header("✅ KYC Process Completed!")
            st.info("Thank you. The result is shown in the KYC Assistant chat.")
            if st.button("Start New Verification"):
                st.session_state.step = 1
                st.session_state.id_image = None
                st.session_state.id_face_encoding = None
                st.session_state.selfie_image = None
                st.session_state.face_match_result = None
                st.session_state.ocr_text = ""
                st.session_state.extracted_form_data = {}
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
                with st.chat_message("user"): st.markdown(user_prompt)
            with st.spinner("Thinking..."):
                model = genai.GenerativeModel(model_name='gemini-1.5-flash', system_instruction=CHATBOT_PROMPT)
                response = model.generate_content(user_prompt)
                agent_response = response.text
                st.session_state.messages.append({"role": "assistant", "content": agent_response})
                st.rerun()