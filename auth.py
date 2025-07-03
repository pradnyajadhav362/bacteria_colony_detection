# auth.py - simplified authentication
# google-style sign-in and email login for streamlit app

import streamlit as st
import os
import json
from datetime import datetime

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
        st.title("Bacterial Colony Analyzer")
        st.write("Choose how to sign in")
        
        st.markdown("---")
        
        # login options tabs
        tab1, tab2 = st.tabs(["Google Sign-In", "Email Login"])
        
        with tab1:
            st.write("Sign in with your Google account")
            
            with st.form("google_login_form"):
                google_email = st.text_input("Google email:", placeholder="")
                google_submit = st.form_submit_button("Continue with Google", type="primary")
                
                if google_submit:
                    if google_email and "@" in google_email:
                        # extract name from email
                        name = google_email.split("@")[0].replace(".", " ").title()
                        
                        # log the access
                        log_user_access(google_email, f"{name} (Google)")
                        
                        # set session state
                        st.session_state.authenticated = True
                        st.session_state.user_email = google_email
                        st.session_state.user_name = f"{name} (Google)"
                        
                        st.success(f"Welcome back, {name}")
                        st.rerun()
                    else:
                        st.error("Please enter a valid Google email address")
        
        with tab2:
            st.write("Enter any email address to access the app")
            st.write("We track usage for analytics but all emails are welcome")
            
            # simple email form
            with st.form("login_form"):
                email = st.text_input("Your email address:", placeholder="")
                name = st.text_input("Your name (optional):", placeholder="")
                submitted = st.form_submit_button("Enter App")
                
                if submitted:
                    if email and "@" in email:
                        # log the access
                        log_user_access(email, name)
                        
                        # set session state
                        st.session_state.authenticated = True
                        st.session_state.user_email = email
                        st.session_state.user_name = name or email.split("@")[0]
                        
                        st.success(f"Welcome, {st.session_state.user_name}")
                        st.rerun()
                    else:
                        st.error("Please enter a valid email address")
        
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
            
            # detailed stats for admin
            st.sidebar.metric("total logins", stats['total_logins'])
            st.sidebar.metric("unique users", stats['unique_users'])
            
            if stats["recent_users"]:
                st.sidebar.markdown("**recent users:**")
                for user in stats["recent_users"][-3:]:  # show last 3 users
                    st.sidebar.write(f"• {user}")
            
            if st.sidebar.button("view full log"):
                st.sidebar.write("check local_files/user_access_log.json")
        
        if st.sidebar.button("Logout"):
            for key in ['authenticated', 'user_email', 'user_name']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

def init_auth():
    # initialize simple email authentication with logging
    authenticate()
    show_user_info()
    return None 