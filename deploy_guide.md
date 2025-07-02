# 🚀 Streamlit Cloud Deployment Guide with Authentication

This guide will help you deploy your Bacterial Colony Analyzer app to Streamlit Cloud with email-based authentication.

## 📋 Prerequisites

1. **GitHub Account**: You'll need a GitHub account to host your code
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Python Environment**: Make sure all dependencies are in `requirements.txt`

## 🔧 Setup Steps

### 1. Prepare Your Repository

Make sure your GitHub repository contains these files:
```
your-repo/
├── app.py                 # Main Streamlit app
├── auth.py               # Authentication module
├── colony_analyzer.py    # Analysis pipeline
├── requirements.txt      # Python dependencies
├── .streamlit/
│   └── config.toml      # Streamlit configuration
├── allowed_emails.txt   # List of authorized emails
└── README.md            # Project documentation
```

### 2. Configure Authentication

#### Option A: Using allowed_emails.txt (Recommended)
Edit `allowed_emails.txt` and add the email addresses of people who should have access:

```txt
# Allowed email addresses for Bacterial Colony Analyzer
# Add one email per line
# Lines starting with # are comments

your-actual-email@example.com
colleague1@company.com
student1@university.edu
researcher@lab.org
```

#### Option B: Using Environment Variables
In Streamlit Cloud, you can set environment variables:
1. Go to your app settings in Streamlit Cloud
2. Add environment variable: `ALLOWED_EMAILS`
3. Value: `email1@example.com,email2@example.com,email3@example.com`

#### Option C: Hardcoded in auth.py
Edit the `get_allowed_emails()` function in `auth.py`:

```python
def get_allowed_emails() -> List[str]:
    return [
        "your-email@example.com",
        "colleague1@example.com",
        # Add more emails here
    ]
```

### 3. Deploy to Streamlit Cloud

1. **Push to GitHub**: Make sure all your code is pushed to a GitHub repository

2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set the main file path to: `app.py`
   - Click "Deploy"

3. **Configure App Settings**:
   - In your app settings, you can set environment variables
   - Add any secret keys or configuration

### 4. Test Authentication

1. Visit your deployed app URL
2. You should see a login page
3. Try logging in with an authorized email
4. Try logging in with an unauthorized email (should be denied)

## 🔒 Security Features

The authentication system includes:

- **Email-based access control**: Only pre-approved emails can access
- **Session management**: Users stay logged in during their session
- **Logout functionality**: Users can log out manually
- **Clean UI**: Professional login interface
- **Error handling**: Clear messages for unauthorized access

## 🛠️ Customization Options

### Change Authentication Method

You can modify `auth.py` to use different authentication methods:

1. **Database Authentication**: Connect to a database of users
2. **OAuth**: Use Google, GitHub, or other OAuth providers
3. **Password Protection**: Add password requirements
4. **Time-based Access**: Limit access to certain time periods

### Add More Security

```python
# In auth.py, you can add:
- Rate limiting for login attempts
- IP address restrictions
- Session timeout
- Audit logging
```

## 📊 Monitoring and Management

### View Active Users
The app shows who is currently logged in in the sidebar.

### Update Access List
To add/remove users:
1. Edit `allowed_emails.txt`
2. Push changes to GitHub
3. Streamlit Cloud will automatically redeploy

### Environment Variables in Streamlit Cloud
You can set these in your app settings:
- `ALLOWED_EMAILS`: Comma-separated list of emails
- `SECRET_KEY`: Custom secret key for sessions

## 🚨 Troubleshooting

### Common Issues

1. **"Access denied" for valid email**:
   - Check email spelling in `allowed_emails.txt`
   - Ensure no extra spaces or characters
   - Verify the file is in your repository

2. **App not deploying**:
   - Check `requirements.txt` has all dependencies
   - Verify `app.py` is the main file
   - Check for syntax errors

3. **Authentication not working**:
   - Ensure `auth.py` is imported correctly
   - Check that `init_auth()` is called in `app.py`
   - Verify the allowed emails list is not empty

### Debug Mode

To debug authentication locally:
```bash
streamlit run app.py --server.headless=false
```

## 🔄 Updating the App

1. Make changes to your code
2. Push to GitHub
3. Streamlit Cloud automatically redeploys
4. Test the new version

## 📞 Support

If you encounter issues:
1. Check the Streamlit Cloud logs
2. Verify all files are in your repository
3. Test locally first with `streamlit run app.py`
4. Check the Streamlit documentation

## 🎯 Next Steps

After deployment, you can:
- Share the app URL with authorized users
- Monitor usage through Streamlit Cloud dashboard
- Add more features like user management
- Implement advanced security measures

---

**Note**: This authentication system is suitable for small to medium teams. For enterprise use, consider implementing more robust authentication with proper user management systems. 