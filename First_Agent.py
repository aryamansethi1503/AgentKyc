import streamlit as st
import google.generativeai as genai
import json
import base64
from datetime import date, datetime
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="KYC Multi-Modal Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SYSTEM_PROMPT = """
You are a friendly and professional KYC (Know Your Customer) agent. 
Your goal is to guide the user through the process of filling out their information and uploading their documents.
- Be conversational and helpful.
- Acknowledge when the user provides information or uploads a file.
- If the user asks for help, provide clear instructions.
- Once all fields are filled and both documents are uploaded, confirm that the KYC process is complete and thank the user.
- Do not ask for the information directly. Guide them to use the form.
- Keep your responses concise.
"""

def call_gemini_api(prompt):
    """Sends a prompt to the Gemini API and returns the response."""
    try:
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=SYSTEM_PROMPT
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred with the AI model: {e}")
        return "Sorry, I'm having trouble connecting. Please check your API key and try again."

def init_session_state():
    """Initializes all the necessary variables in the session state."""
    if 'api_key_configured' not in st.session_state:
        st.session_state.api_key_configured = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "form_data" not in st.session_state:
        st.session_state.form_data = {
            "fullName": "",
            "dob": None,
            "address": "",
            "idCard": None,
            "proofOfAddress": None
        }

    if "greeted" not in st.session_state:
        st.session_state.greeted = False

init_session_state()

api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    st.title("API Key Missing")
    st.warning("The API key is not available. Please ensure the API key is set in the `.env` file.")
else:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        st.session_state.api_key_configured = True

        if not st.session_state.greeted:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Hello! I'm here to help you with the KYC process. Please fill in your details in the form and upload the required documents."
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
            full_name = st.text_input("Full Name", key="fullName")
            dob = st.date_input("Date of Birth", min_value=date(1920, 1, 1), key="dob")
            address = st.text_area("Full Address", key="address")

            st.markdown("---")
            st.subheader("Document Upload")
            id_card = st.file_uploader(
                "ID Card (Driver's License, etc.)",
                type=['jpg', 'jpeg', 'png', 'pdf'],
                key="idCard"
            )
            proof_of_address = st.file_uploader(
                "Proof of Address (Utility Bill)",
                type=['jpg', 'jpeg', 'png', 'pdf'],
                key="proofOfAddress"
            )

            submitted = st.form_submit_button(
                "Submit KYC", 
                use_container_width=True,
                type="primary"
            )

            if submitted:
                if not all([full_name, dob, address, id_card, proof_of_address]):
                    st.warning("Please fill out all fields and upload both documents before submitting.")
                    system_notification = "User tried to submit the form, but it's incomplete. Please remind them to fill all fields and upload both documents."
                    agent_response = call_gemini_api(system_notification)
                    st.session_state.messages.append({"role": "assistant", "content": agent_response})
                else:
                    try:
                        id_card_b64 = base64.b64encode(id_card.getvalue()).decode('utf-8')
                        proof_addr_b64 = base64.b64encode(proof_of_address.getvalue()).decode('utf-8')
                        
                        kyc_data = {
                            "fullName": full_name,
                            "dob": dob.isoformat(),
                            "address": address,
                            "documents": {
                                "idCard": {
                                    "name": id_card.name,
                                    "type": id_card.type,
                                    "data": id_card_b64,
                                },
                                "proofOfAddress": {
                                    "name": proof_of_address.name,
                                    "type": proof_of_address.type,
                                    "data": proof_addr_b64,
                                }
                            },
                            "submissionDate": datetime.now().isoformat()
                        }

                        with open("kyc_data.json", "w") as f:
                            json.dump(kyc_data, f, indent=4)
                        
                        st.success("KYC data submitted successfully and saved to `kyc_data.json`!")
            
                        system_notification = "User has successfully submitted the complete form. Confirm and thank them."
                        agent_response = call_gemini_api(system_notification)
                        st.session_state.messages.append({"role": "assistant", "content": agent_response})

                    except Exception as e:
                        st.error(f"Error saving data: {e}")
                        system_notification = "There was an error saving the user's data. Please inform the user and ask them to try again."
                        agent_response = call_gemini_api(system_notification)
                        st.session_state.messages.append({"role": "assistant", "content": agent_response})

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
                agent_response = call_gemini_api(user_prompt)
                st.session_state.messages.append({"role": "assistant", "content": agent_response})
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(agent_response)
            st.rerun()
