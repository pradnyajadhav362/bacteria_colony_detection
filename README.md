#  Bacterial Colony Analysis Pipeline

A comprehensive image analysis pipeline for detecting, characterizing, and scoring bacterial colonies from petri dish images. This project includes both a modular Python class for analysis and a Streamlit web application with authentication.

##  Features

### Analysis Pipeline
- **Image Preprocessing**: Noise reduction, contrast enhancement, gamma correction
- **Plate Detection**: Automatic detection of petri dish boundaries
- **Colony Segmentation**: Advanced algorithms to separate individual colonies
- **Morphology Analysis**: Size, shape, roundness, and edge characteristics
- **Color Clustering**: Group colonies by visual similarity
- **Density Analysis**: Opacity, texture, and density measurements
- **Scoring System**: Combined interest scores for colony prioritization

### Web Application
- **Interactive Interface**: User-friendly Streamlit app
- **Parameter Control**: Adjustable analysis parameters
- **Visualization**: Multiple result views and plots
- **Authentication**: Email-based access control
- **Cloud Deployment**: Ready for Streamlit Cloud

##  Authentication

The app includes email-based authentication to control access:

### Setup Access List

#### For Local Development:
Create `local_files/allowed_emails.txt` and add authorized email addresses:
```txt
prjadhav@andrew.cmu.edu
colleague1@company.com
student1@university.edu
```

#### For Streamlit Cloud Deployment:
1. Go to your app in Streamlit Cloud
2. Click "Settings" → "Secrets"
3. Add environment variable:
```toml
ALLOWED_EMAILS = "prjadhav@andrew.cmu.edu,colleague1@company.com,student1@university.edu"
```

### Authentication Features
- Email-based login
- Session management
- Logout functionality
- Access control for authorized users only

##  Installation

### Local Development
```bash
# Clone the repository
git clone <your-repo-url>
cd bacterial-colony-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py
```

### Dependencies
- Python 3.8+
- OpenCV
- scikit-image
- scikit-learn
- pandas
- matplotlib
- plotly
- streamlit

##  Deployment

### Streamlit Cloud Deployment
1. Push your code to GitHub
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file to `app.py`
5. Deploy

See `deploy_guide.md` for detailed deployment instructions.

##  Usage

### Web Application
1. **Login**: Enter your authorized email address
2. **Upload Image**: Select a petri dish image
3. **Adjust Parameters**: Modify analysis settings if needed
4. **Run Analysis**: Click "Run Analysis" to process
5. **Explore Results**: View results in different tabs

### Admin Dashboard
Access comprehensive user data and analytics:

```bash
python run_admin.py
```

**Admin Features:**
- View all user uploads and analysis results
- Track usage statistics and colony detection metrics
- Browse individual sessions with full details
- Export all data for analysis or backup
- Download original images, processed images, and colony data
- Monitor analysis parameters and success rates

**Admin Login:** Set secure password via environment variable (see `admin_setup.md`)

### Programmatic Usage
```python
from colony_analyzer import ColonyAnalyzer

# Initialize analyzer
analyzer = ColonyAnalyzer()

# Run full analysis
results = analyzer.run_full_analysis("path/to/image.jpg")

# Access results
print(f"Found {len(results['colony_properties'])} colonies")
print(f"Top colony score: {results['top_colonies']['bio_interest'].max()}")
```

##  Project Structure

```
bacterial-colony-analyzer/
 app.py                 # Main Streamlit application
 auth.py               # Authentication module
 colony_analyzer.py    # Core analysis pipeline
 requirements.txt      # Python dependencies
 allowed_emails.txt   # Authorized email addresses
 .streamlit/
    config.toml      # Streamlit configuration
 deploy_guide.md      # Deployment instructions
 test_colony_analyzer.py  # Test suite
 demo_data_generator.py   # Sample data generator
 README.md            # This file
```

##  Configuration

### Analysis Parameters
- **Preprocessing**: Bilateral filtering, CLAHE, gamma correction
- **Segmentation**: Adaptive thresholding, watershed, size filters
- **Clustering**: K-means color clustering
- **Scoring**: Morphology, density, and novelty weights

### Authentication Options
- **File-based**: Edit `allowed_emails.txt`
- **Environment Variables**: Set `ALLOWED_EMAILS`
- **Hardcoded**: Modify `auth.py` directly

##  Results

The analysis provides:
- **Colony Count**: Total number of detected colonies
- **Morphology Data**: Size, shape, and edge characteristics
- **Color Groups**: Clustered colonies by visual similarity
- **Density Metrics**: Opacity and texture measurements
- **Interest Scores**: Combined scoring for colony prioritization
- **Visualizations**: Interactive plots and annotated images

## 🧪 Testing

Run the test suite:
```bash
python test_colony_analyzer.py
```

Generate demo data:
```bash
python demo_data_generator.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Support

For issues and questions:
1. Check the troubleshooting section in `deploy_guide.md`
2. Review the test suite for usage examples
3. Open an issue on GitHub

##  Scientific Applications

This pipeline is designed for:
- Microbiology research
- Colony counting and characterization
- High-throughput screening
- Quality control in laboratory settings
- Educational purposes

---

**Note**: This tool is designed for research and educational use. Always validate results with manual inspection for critical applications. 