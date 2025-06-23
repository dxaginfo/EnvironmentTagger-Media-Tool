# EnvironmentTagger Media Automation Tool

## Overview
EnvironmentTagger is an AI-powered media tagging tool that automatically identifies and tags environment elements in video and image content. Using Google Cloud Vision API and Gemini API, it can recognize settings, scenery, props, and environmental conditions to create detailed metadata that improves searchability and organization of media assets.

## Features
- Automatic detection of indoor/outdoor settings
- Classification of environment elements (furniture, nature, architecture)
- Recognition of lighting conditions and time of day
- Contextual understanding of environments using Gemini API
- Hierarchical tagging system with confidence scores
- RESTful API for integration with media workflows

## Getting Started

### Prerequisites
- Google Cloud account with Vision API enabled
- Cloud Storage bucket for metadata storage
- Python 3.9+

### Setup
1. Clone this repository
```bash
git clone https://github.com/dxaginfo/EnvironmentTagger-Media-Tool.git
cd EnvironmentTagger-Media-Tool
```

2. Set up Python environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set environment variables
```bash
export PROJECT_ID="your-gcp-project-id"
export BUCKET_NAME="your-storage-bucket"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

4. Run the application
```bash
python app.py
```

### Deployment to Google Cloud

1. Build and deploy as a Cloud Run service
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/environment-tagger
gcloud run deploy environment-tagger --image gcr.io/PROJECT_ID/environment-tagger --platform managed
```

2. Alternatively, deploy as a Cloud Function
```bash
gcloud functions deploy tag_media --runtime python39 --trigger-http --entry-point tag_media
```

## API Reference

### POST /api/tag
Tags a media file with environment metadata
```json
{
  "mediaUrl": "https://example.com/image.jpg",
  "options": {
    "detailLevel": "detailed",
    "includeConfidence": true,
    "customTags": ["office", "modern"]
  }
}
```

### GET /api/tags/{mediaId}
Retrieves environment tags for a specific media

### PUT /api/tags/{mediaId}
Updates environment tags for a specific media
```json
{
  "tags": ["office", "modern", "bright"],
  "override": false
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- Google Cloud Vision API
- Gemini API for AI-powered analysis
- Flask for the web framework