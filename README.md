# EnvironmentTagger Media Tool

EnvironmentTagger is an intelligent media analysis tool that automatically detects, categorizes, and tags environments and settings within media assets. Using computer vision and Gemini API, it identifies locations, lighting conditions, ambient settings, and other environmental factors to improve searchability and production continuity.

## Features

- Automatic environment detection and tagging
- Categorization by location, lighting, weather, time-of-day, and mood
- Integration with Google Cloud Vision and Gemini API
- RESTful API for easy integration
- Batch processing capabilities
- Custom tag dictionary support
- Firestore storage for tag persistence

## Getting Started

### Prerequisites

- Google Cloud Platform account
- Python 3.9 or higher
- Google Cloud SDK installed and configured
- Required API permissions:
  - Gemini API
  - Cloud Vision API
  - Cloud Storage
  - Firestore
  - (Optional) Cloud Functions for serverless deployment

### Installation

```bash
# Clone the repository
git clone https://github.com/dxaginfo/EnvironmentTagger-Media-Tool.git
cd EnvironmentTagger-Media-Tool

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure Google Cloud credentials
gcloud auth application-default login
```

### Local Development

```bash
# Run the Flask development server
python app.py
```

### Deployment Options

#### Google Cloud Functions

```bash
gcloud functions deploy environment-tagger \
  --runtime python39 \
  --trigger-http \
  --allow-unauthenticated \
  --entry-point process_media \
  --memory 2GB \
  --timeout 540s
```

#### Docker Container

```bash
docker build -t environment-tagger:latest .
docker run -p 8080:8080 environment-tagger:latest
```

## API Usage

### Analyze a media file

```bash
curl -X POST https://your-function-url/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "mediaSource": "gs://your-bucket/your-media-file.jpg",
    "analysisOptions": {
      "detectionThreshold": 0.75,
      "maxTags": 20,
      "outputFormat": "json"
    }
  }'
```

### Process multiple files in batch

```bash
curl -X POST https://your-function-url/batch \
  -H "Content-Type: application/json" \
  -d '{
    "mediaSources": [
      "gs://your-bucket/file1.jpg",
      "gs://your-bucket/file2.jpg"
    ],
    "analysisOptions": {
      "detectionThreshold": 0.7
    }
  }'
```

### Retrieve previously generated tags

```bash
curl -X GET https://your-function-url/tags/7e5d82f1a3e245a9b0e9f1a3e245a9b0
```

## Integration with Other Tools

EnvironmentTagger is designed to work seamlessly with other media automation tools:

- **SceneValidator**: Use environmental context to validate scene consistency
- **TimelineAssembler**: Ensure environment consistency across a timeline
- **Google Drive**: Tag and organize media assets based on environmental context

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Google Cloud Vision API
- Google Gemini API
- TensorFlow project
- Flask framework