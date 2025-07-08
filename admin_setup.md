# Admin Setup Guide

## Security Notice

The admin password is **NOT stored in the code** for security reasons. You must set it as an environment variable.

## Local Development Setup

### Option 1: Environment Variable
```bash
# Set admin password for current session
export ADMIN_PASSWORD="your_secure_password_here"

# Then run the app
python3 app.py
```

### Option 2: .env File (Recommended)
```bash
# Create .env file (this file is git-ignored)
echo "ADMIN_PASSWORD=your_secure_password_here" > .env

# Install python-dotenv if not already installed
pip install python-dotenv

# The app will automatically load from .env file
python3 app.py
```

## Production/Cloud Deployment Setup

### Streamlit Cloud
1. Go to your app settings in Streamlit Cloud
2. Navigate to "Secrets" section
3. Add this environment variable:
```toml
ADMIN_PASSWORD = "your_secure_password_here"
```

### Heroku
```bash
heroku config:set ADMIN_PASSWORD="your_secure_password_here"
```

### AWS/Other Cloud
Set environment variable in your deployment configuration:
```
ADMIN_PASSWORD=your_secure_password_here
```

## Access Admin Dashboard

### In Main App
1. Go to your deployed app
2. Select "Admin Dashboard" mode
3. Enter your secure admin password
4. Access all user data and analytics

### Separate Admin Dashboard
```bash
# Set password first
export ADMIN_PASSWORD="your_secure_password_here"

# Run admin dashboard on separate port
python3 run_admin.py
```

## Default Password

If no environment variable is set, the default password is: `default_admin_123`

**⚠️ IMPORTANT: Change this immediately in production!**

## What Admin Access Provides

- View all user uploads and images
- See analysis results and parameters for every session
- Download colony data CSVs for all users
- Track usage statistics and metrics
- Export complete datasets for research
- Monitor app performance and user behavior

## Security Best Practices

1. **Use a strong password** (12+ characters, mix of letters/numbers/symbols)
2. **Never commit passwords to git**
3. **Use different passwords for development vs production**
4. **Regularly rotate admin passwords**
5. **Limit admin access to necessary personnel only**
6. **Monitor admin access logs**

## Troubleshooting

**Problem:** Admin login not working
**Solution:** Check that `ADMIN_PASSWORD` environment variable is set correctly

**Problem:** Can't see user data
**Solution:** Users must upload images and run analysis first to generate data

**Problem:** Admin dashboard won't start
**Solution:** Make sure port 8502 is available, or modify port in `run_admin.py` 