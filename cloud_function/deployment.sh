#!/bin/bash

# Deploy EnvironmentTagger as a Cloud Function

echo "Deploying EnvironmentTagger as a Cloud Function..."

# Create a temporary deployment directory
rm -rf deploy_tmp
mkdir -p deploy_tmp

# Copy the necessary files
cp -r ../environment_tagger.py deploy_tmp/
cp main.py deploy_tmp/
cp requirements.txt deploy_tmp/

# Deploy the function
gcloud functions deploy environment-tagger \
  --gen2 \
  --runtime=python39 \
  --region=us-central1 \
  --source=deploy_tmp \
  --entry-point=environment_tagger_http \
  --trigger-http \
  --allow-unauthenticated \
  --memory=2048MB \
  --timeout=540s \
  --max-instances=10

# Clean up
rm -rf deploy_tmp

echo "Deployment complete!"
echo "Function URL: https://us-central1-$(gcloud config get-value project).cloudfunctions.net/environment-tagger"