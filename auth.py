# auth.py - Google OAuth Authentication
# secure google sign-in for streamlit app

import streamlit as st
import os
from typing import List
import json
from datetime import datetime

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    import google.auth
except ImportError:
    st.error("google auth libraries not installed. run: pip install google-auth google-auth-oauthlib")
    st.stop()

def get_allowed_emails() -> List[str]:
    # get list of allowed email addresses
    # priority 1: streamlit secrets
    # priority 2: environment variable  
    # priority 3: local file
    
    # option 1: read from streamlit secrets
    try:
        if hasattr(st, 'secrets') and st.secrets:
            allowed_emails = st.secrets.get("auth", {}).get("allowed_emails", [])
            if allowed_emails:
                return allowed_emails
    except Exception:
        pass
    
    # option 2: read from environment variable
    emails_str = os.getenv("ALLOWED_EMAILS", "")
    if emails_str:
        allowed_emails = []
        for email in emails_str.split(","):
            email = email.strip()
            if email and "@" in email:
                allowed_emails.append(email)
        if allowed_emails:
            return allowed_emails
    
    # option 3: read from local file
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
    
    return []

def google_oauth_flow():
    # simplified google oauth for streamlit
    client_id = os.getenv("GOOGLE_CLIENT_ID") or st.secrets.get("google", {}).get("client_id", "")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET") or st.secrets.get("google", {}).get("client_secret", "")
    
    if not client_id or not client_secret:
        st.error("google oauth not configured. need GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET")
        st.info("get these from google cloud console")
        return None
    
    # create oauth flow
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost:8501"]
            }
        },
        scopes=["openid", "email", "profile"]
    )
    
    flow.redirect_uri = "http://localhost:8501"
    return flow

def log_user_access(email: str, name: str = ""):
    # log user access to local file for tracking
    log_entry = {
        "email": email,
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "session_id": st.session_state.get("session_id", "unknown")
    }
    
    # append to log file
    log_file = "local_files/user_access_log.json"
    try:
        # ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # read existing logs
        logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read().strip()
                if content:
                    logs = json.loads(content)
        
        # add new log entry
        logs.append(log_entry)
        
        # save back to file
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
            
        print(f"logged access: {email} at {log_entry['timestamp']}")
    except Exception as e:
        print(f"could not log user access: {e}")

def get_user_stats():
    # get usage statistics from log file
    log_file = "local_files/user_access_log.json"
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.loads(f.read())
            
            # count unique users and total logins
            unique_emails = set()
            total_logins = len(logs)
            
            for log in logs:
                unique_emails.add(log.get("email", "unknown"))
            
            return {
                "total_logins": total_logins,
                "unique_users": len(unique_emails),
                "recent_users": list(unique_emails)[-5:]  # last 5 unique users
            }
    except Exception:
        pass
    
    return {"total_logins": 0, "unique_users": 0, "recent_users": []}

def authenticate():
    # simple email authentication that accepts any email
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not st.session_state.authenticated:
        st.title("bacterial colony analyzer")
        st.write("enter your email to access the app")
        st.write("we track usage for analytics but all emails are welcome")
        
        # show usage stats
        stats = get_user_stats()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("total logins", stats["total_logins"])
        with col2:
            st.metric("unique users", stats["unique_users"])
        with col3:
            if stats["recent_users"]:
                st.write("recent users:")
                for user in stats["recent_users"]:
                    st.write(f"• {user}")
        
        st.markdown("---")
        
        # simple email form
        with st.form("login_form"):
            email = st.text_input("your email address:", placeholder="name@company.com")
            name = st.text_input("your name (optional):", placeholder="john doe")
            submitted = st.form_submit_button("enter app")
            
            if submitted:
                if email and "@" in email:
                    # log the access
                    log_user_access(email, name)
                    
                    # set session state
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.session_state.user_name = name or email.split("@")[0]
                    
                    st.success(f"welcome, {st.session_state.user_name}!")
                    st.rerun()
                else:
                    st.error("please enter a valid email address")
        
        st.stop()
    
    return st.session_state.authenticated

def show_user_info():
    # display current user info and admin stats
    if 'user_email' in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**logged in as:** {st.session_state.get('user_name', 'user')}")
        st.sidebar.markdown(f"**email:** {st.session_state.user_email}")
        
        # show admin stats if user email matches certain pattern
        if st.session_state.user_email == "prjadhav@andrew.cmu.edu":
            st.sidebar.markdown("---")
            st.sidebar.markdown("**admin panel:**")
            stats = get_user_stats()
            st.sidebar.write(f"total logins: {stats['total_logins']}")
            st.sidebar.write(f"unique users: {stats['unique_users']}")
            
            if st.sidebar.button("view full log"):
                st.sidebar.write("check local_files/user_access_log.json")
        
        if st.sidebar.button("🚪 logout"):
            for key in ['authenticated', 'user_email', 'user_name']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

def init_auth():
    # initialize simple email authentication with logging
    authenticate()
    show_user_info()
    return None 