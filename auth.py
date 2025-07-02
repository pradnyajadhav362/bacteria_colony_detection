# auth.py - Minimal implementation for both local and cloud deployment
# This file is public on GitHub but contains no sensitive data

import streamlit as st
import os
from typing import List

def get_allowed_emails() -> List[str]:
    # Get list of allowed email addresses
    # Priority 1: Environment variable (for Streamlit Cloud deployment)
    # Priority 2: Local file (for local development)
    
    # Option 1: Read from environment variable (recommended for deployment)
    emails_str = os.getenv("ALLOWED_EMAILS", "")
    if emails_str:
        allowed_emails = [email.strip() for email in emails_str.split(",") if email.strip()]
        if allowed_emails:
            return allowed_emails
    
    # Option 2: Read from local file (for local development)
    try:
        with open("local_files/allowed_emails.txt", "r") as f:
            allowed_emails = []
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    allowed_emails.append(line)
            if allowed_emails:
                return allowed_emails
    except FileNotFoundError:
        pass
    
    # No hardcoded fallback - user must set up environment variable or local file
    return []

def init_auth():
    # Initialize authentication system
    # Try to import from local_files first (for local development)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("local_auth", "local_files/auth.py")
        if spec and spec.loader:
            local_auth = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(local_auth)
            return local_auth.init_auth()
    except (ImportError, FileNotFoundError, Exception):
        pass
    
    # Fallback to minimal implementation for cloud deployment
    return _minimal_auth()

def _minimal_auth():
    # Minimal authentication for cloud deployment
    allowed_emails = get_allowed_emails()
    
    if not allowed_emails:
        st.error("No allowed emails configured. Please set ALLOWED_EMAILS environment variable.")
        st.stop()
    
    # Simple authentication check
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.markdown("""
        # Bacterial Colony Analyzer - Authentication Required
        
        This application requires authentication to access.
        Please enter your email address to continue.
        """)
        
        with st.form("login_form"):
            email = st.text_input("Email Address", placeholder="Enter your email address")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if email.lower().strip() in [e.lower().strip() for e in allowed_emails]:
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.success(f"Welcome, {email}!")
                    st.rerun()
                else:
                    st.error(" Access denied. This email is not authorized to use this application.")
                    st.info("If you believe you should have access, please contact the administrator.")
        
        st.stop()
    
    # Show user info
    if 'user_email' in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"** Logged in as:** {st.session_state.user_email}")
        if st.sidebar.button(" Logout"):
            if 'authenticated' in st.session_state:
                del st.session_state.authenticated
            if 'user_email' in st.session_state:
                del st.session_state.user_email
            st.rerun()
    
    return None  # No auth object needed for minimal implementation 