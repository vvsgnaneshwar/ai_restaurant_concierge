import os
import streamlit as st
from utils import get_reply_to_customer

# --- CUSTOM CSS FOR STYLING ---
def inject_custom_css():
    """
    Injects custom CSS to style the Streamlit app with a sophisticated gray-black gradient.
    """
    st.markdown(
        """
        <style>
            /* Import Google Fonts for better typography */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            /* Main app background with sophisticated gradient */
            [data-testid="stApp"] {
                background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 25%, #404040 50%, #2d2d2d 75%, #1a1a1a 100%);
                background-attachment: fixed;
                color: #E5E5E5;
                font-family: 'Inter', sans-serif;
            }

            /* Main container styling */
            .main > div {
                padding: 2rem 1rem;
                background: rgba(255, 255, 255, 0.02);
                border-radius: 20px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                margin: 1rem 0;
            }

            /* Header and titles with improved styling */
            h1 {
                color: #FFFFFF;
                font-weight: 600;
                text-align: center;
                margin-bottom: 0.5rem;
                text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
            }
            
            h2, h3 {
                color: #F0F0F0;
                font-weight: 500;
                text-align: center;
                margin-bottom: 2rem;
                opacity: 0.9;
            }
            
            h4, h5, h6 {
                color: #E0E0E0;
                font-weight: 400;
            }
            
            /* Enhanced chat message bubbles */
            [data-testid="stChatMessage"] {
                background: linear-gradient(145deg, rgba(60, 60, 60, 0.4), rgba(40, 40, 40, 0.6));
                border-radius: 15px;
                padding: 1.5rem;
                margin: 1rem 0;
                border: 1px solid rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(5px);
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                transition: transform 0.2s ease;
            }
            
            [data-testid="stChatMessage"]:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
            }

            /* User messages - slightly different styling */
            [data-testid="stChatMessage"][data-testid*="user"] {
                background: linear-gradient(145deg, rgba(70, 130, 180, 0.3), rgba(100, 149, 237, 0.2));
                border-left: 3px solid #4682B4;
            }

            /* Assistant messages */
            [data-testid="stChatMessage"][data-testid*="assistant"] {
                background: linear-gradient(145deg, rgba(50, 80, 50, 0.3), rgba(60, 100, 60, 0.2));
                border-left: 3px solid #32CD32;
            }

            /* Chat input styling */
            [data-testid="stChatInput"] {
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                padding: 0 !important;
            }
            
            [data-testid="stChatInput"] input {
                background: transparent;
                color: #E5E5E5;
                border: none;
                font-size: 16px;
                padding: 12px 16px;
            }
            
            [data-testid="stChatInput"] input::placeholder {
                color: #888888;
                font-style: italic;
            }

            /* Spinner styling */
            .stSpinner > div {
                border-top-color: #4682B4 !important;
            }

            /* Button styling */
            .stButton > button {
                background: linear-gradient(145deg, #4682B4, #5A9BD4);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 0.5rem 1rem;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                background: linear-gradient(145deg, #5A9BD4, #4682B4);
                box-shadow: 0 4px 15px rgba(70, 130, 180, 0.4);
                transform: translateY(-2px);
            }

            /* Error message styling */
            .stAlert {
                background: linear-gradient(145deg, rgba(220, 53, 69, 0.2), rgba(220, 53, 69, 0.1));
                border: 1px solid rgba(220, 53, 69, 0.3);
                border-radius: 10px;
                color: #FFB3BA;
            }

            /* Sidebar styling (if used) */
            .css-1d391kg {
                background: linear-gradient(180deg, rgba(30, 30, 30, 0.9), rgba(20, 20, 20, 0.95));
                border-right: 1px solid rgba(255, 255, 255, 0.1);
            }

            /* Footer and caption styling */
            .st-emotion-cache-1cypcdb,
            [data-testid="stCaption"] {
                color: #888888;
                text-align: center;
                font-size: 0.9rem;
                font-weight: 300;
            }

            /* Horizontal rule styling */
            hr {
                border: none;
                height: 1px;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
                margin: 2rem 0;
            }

            /* Avatar styling improvements */
            [data-testid="stChatMessage"] img {
                border-radius: 50%;
                border: 2px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            }

            /* Scrollbar styling */
            ::-webkit-scrollbar {
                width: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: rgba(40, 40, 40, 0.5);
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(180deg, #555, #333);
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(180deg, #666, #444);
            }

            /* Animation for new messages */
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            [data-testid="stChatMessage"]:last-child {
                animation: slideIn 0.3s ease-out;
            }

            /* Remove bottom container padding */
            [data-testid="stBottomBlockContainer"] {
                padding-bottom: 20px !important;
                margin-bottom: 0 !important;
            }

        </style>
        """,
        unsafe_allow_html=True
    )

# --- PAGE CONFIGURATION ---
# Must be the first Streamlit command
st.set_page_config(
    page_title="Nom Nom",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply the custom CSS
inject_custom_css()

# --- SESSION STATE INITIALIZATION ---
# Initialize conversation history for the backend logic.
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = ""

# Initialize chat log for displaying on the UI.
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# --- UI ELEMENTS ---
st.title("🍔 Nom Nom, your GoodFoods Concierge")
st.subheader("Your AI-powered guide to our restaurant venues & reservations")

# Add some visual spacing
# st.markdown("<br>", unsafe_allow_html=True)

# Add an initial assistant message if the chat is new
if not st.session_state.chat_log:
    initial_message = "Hello! I'm Nom Nom, your personal assistant for GoodFoods. How can I help you today?"
    st.session_state.chat_log.append(("Assistant", initial_message))
    # Update the backend history with this first message
    st.session_state.conversation_history = f"Assitant to Customer: {initial_message}\n"

# Display chat history from the log
for sender, message in st.session_state.chat_log:
    avatar = "👤" if sender == "Customer" else "🤖"
    with st.chat_message(sender, avatar=avatar):
        st.write(message)

# --- CHAT INPUT & PROCESSING ---
if user_input := st.chat_input("Ask about reservations, menus, locations, or anything else..."):
    # 1. Add user message to the chat log and display it
    st.session_state.chat_log.append(("Customer", user_input))
    with st.chat_message("Customer", avatar="👤"):
        st.write(user_input)

    # 2. Show a thinking spinner and get the assistant's reply
    with st.chat_message("Assistant", avatar="🤖"):
        with st.spinner("Nom Nom is thinking..."):
            try:
                # Call the backend to get a response
                reply, updated_history = get_reply_to_customer(
                    st.session_state.conversation_history, user_input
                )
                st.write(reply)

                # 3. Update conversation history and log the reply for the next turn
                st.session_state.conversation_history = updated_history
                st.session_state.chat_log.append(("Assistant", reply))

            except Exception as e:
                error_message = f"Sorry, I encountered a technical issue. Please try again later. 🔧"
                st.error(error_message)
                st.session_state.chat_log.append(("Assistant", error_message))
                # Log the actual error for debugging (you might want to use proper logging)
                print(f"Backend error: {e}")