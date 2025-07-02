# auth.py
# email-based authentication for streamlit app

import streamlit as st
import hashlib
import hmac
import time
from typing import List, Optional

class EmailAuth:
    def __init__(self, allowed_emails: List[str], secret_key: str = "your-secret-key"):
        # Initialize email-based authentication
        # Args:
        #     allowed_emails: List of email addresses that are allowed to access the app
        #     secret_key: Secret key for session management
        self.allowed_emails = [email.lower().strip() for email in allowed_emails]
        self.secret_key = secret_key
        
    def check_auth(self) -> bool:
        # Check if user is authenticated
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        return st.session_state.authenticated
    
    def login_page(self) -> bool:
        # Display login page and return True if authentication successful
        st.markdown("""
        # Bacterial Colony Analyzer - Authentication Required
        
        This application requires authentication to access.
        Please enter your email address to continue.
        """)
        
        with st.form("login_form"):
            email = st.text_input("Email Address", placeholder="Enter your email address")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if self.authenticate_user(email):
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.success(f"Welcome, {email}!")
                    st.rerun()
                else:
                    st.error(" Access denied. This email is not authorized to use this application.")
                    st.info("If you believe you should have access, please contact the administrator.")
        
        return False
    
    def authenticate_user(self, email: str) -> bool:
        # Authenticate user based on email
        if not email:
            return False
        
        email_clean = email.lower().strip()
        return email_clean in self.allowed_emails
    
    def logout(self):
        # Logout user
        if 'authenticated' in st.session_state:
            del st.session_state.authenticated
        if 'user_email' in st.session_state:
            del st.session_state.user_email
    
    def show_user_info(self):
        # Display current user information
        if 'user_email' in st.session_state:
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"** Logged in as:** {st.session_state.user_email}")
            if st.sidebar.button(" Logout"):
                self.logout()
                st.rerun()

def get_allowed_emails() -> List[str]:
    # Get list of allowed email addresses
    # You can modify this function to:
    # 1. Read from a file
    # 2. Connect to a database
    # 3. Use environment variables
    # 4. Or hardcode the list
    
    # Option 1: Read from file (recommended)
    try:
        with open("allowed_emails.txt", "r") as f:
            allowed_emails = []
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    allowed_emails.append(line)
            if allowed_emails:
                return allowed_emails
    except FileNotFoundError:
        pass
    
    # Option 2: Read from environment variable
    # import os
    # emails_str = os.getenv("ALLOWED_EMAILS", "")
    # if emails_str:
    #     allowed_emails = [email.strip() for email in emails_str.split(",") if email.strip()]
    #     if allowed_emails:
    #         return allowed_emails
    
    # Option 3: Hardcoded fallback
    allowed_emails = [
        "your-email@example.com",
        "colleague1@example.com", 
        "colleague2@example.com",
        "student1@university.edu",
        "researcher@lab.org"
    ]
    
    return allowed_emails

def init_auth():
    # Initialize authentication system
    allowed_emails = get_allowed_emails()
    auth = EmailAuth(allowed_emails)
    
    if not auth.check_auth():
        auth.login_page()
        st.stop()
    
    auth.show_user_info()
    return auth 