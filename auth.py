# auth.py - Streamlit Secrets Authentication
# This file is public on GitHub but contains no sensitive data

import streamlit as st
import os
from typing import List

def get_allowed_emails() -> List[str]:
    # Get list of allowed email addresses
    # Priority 1: Streamlit secrets (recommended for deployment)
    # Priority 2: Environment variable (fallback)
    # Priority 3: Local file (for local development)
    # Priority 4: Simple fallback for testing
    
    # Option 1: Read from Streamlit secrets (recommended)
    try:
        if hasattr(st, 'secrets') and st.secrets:
            allowed_emails = st.secrets.get("auth", {}).get("allowed_emails", [])
            if allowed_emails:
                return allowed_emails
    except Exception as e:
        st.write(f"Debug: Could not read from secrets: {e}")
    
    # Option 2: Read from environment variable (fallback)
    emails_str = os.getenv("ALLOWED_EMAILS", "")
    if emails_str:
        allowed_emails = [email.strip() for email in emails_str.split(",") if email.strip()]
        if allowed_emails:
            return allowed_emails
    
    # Option 3: Read from local file (for local development)
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
    
    # Option 4: No hardcoded fallback for security
    return []

def authenticate():
    # Simple authentication function using Streamlit secrets
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("Bacterial Colony Analyzer - Authentication Required")
        st.write("This application requires authentication to access.")
        st.write("Please enter your email address to continue.")
        
        email = st.text_input("Enter your email:")
        
        if st.button("Login"):
            allowed_emails = get_allowed_emails()
            if not allowed_emails:
                st.error("No allowed emails configured. Please set up authentication.")
                st.stop()
            
            # Normalize email for comparison (lowercase and strip whitespace)
            email_clean = email.lower().strip()
            allowed_emails_clean = [e.lower().strip() for e in allowed_emails]
            
            if email_clean in allowed_emails_clean:
                st.session_state.authenticated = True
                st.session_state.user_email = email
                st.success(f"Welcome, {email}!")
                st.rerun()
            else:
                st.error("Access denied. This email is not authorized to use this application.")
                st.info("If you believe you should have access, please contact the administrator.")
        
        st.stop()
    
    return st.session_state.authenticated

def show_user_info():
    # Display current user information
    if 'user_email' in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"** Logged in as:** {st.session_state.user_email}")
        if st.sidebar.button(" Logout"):
            if 'authenticated' in st.session_state:
                del st.session_state.authenticated
            if 'user_email' in st.session_state:
                del st.session_state.user_email
            st.rerun()

def init_auth():
    # Initialize authentication system
    # Try to import from local_files first (for local development with full features)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("local_auth", "local_files/auth.py")
        if spec and spec.loader:
            local_auth = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(local_auth)
            return local_auth.init_auth()
    except (ImportError, FileNotFoundError, Exception):
        pass
    
    # Fallback to simple authentication for cloud deployment
    authenticate()
    show_user_info()
    return None 