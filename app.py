import os
from flask import Flask, request, jsonify
from environment_tagger import EnvironmentTagger

app = Flask(__name__)

# Initialize tagger with default configuration
tagger = EnvironmentTagger()

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze media and tag environments."""
    request_json = request.get_json(silent=True)
    
    if not request_json or 'mediaSource' not in request_json:
        return jsonify({"error": "Invalid request: mediaSource is required"}), 400
        
    media_source = request_json.get("mediaSource")
    analysis_options = request_json.get("analysisOptions", {})
    
    try:
        # Process the media
        results = tagger.process_media(media_source, analysis_options)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/tags/<media_id>', methods=['GET'])
def get_tags(media_id):
    """Retrieve previously generated tags for a media ID."""
    from google.cloud.firestore import Client as FirestoreClient
    
    try:
        # Initialize Firestore client
        firestore_client = FirestoreClient()
        
        # Get document from Firestore
        doc_ref = firestore_client.collection("environment_tags").document(media_id)
        doc = doc_ref.get()
        
        if doc.exists:
            return jsonify(doc.to_dict())
        else:
            return jsonify({"error": "Tags not found for media ID"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_analyze():
    """Process multiple media files in a batch."""
    request_json = request.get_json(silent=True)
    
    if not request_json or 'mediaSources' not in request_json:
        return jsonify({"error": "Invalid request: mediaSources is required"}), 400
        
    media_sources = request_json.get("mediaSources")
    analysis_options = request_json.get("analysisOptions", {})
    
    if not isinstance(media_sources, list):
        return jsonify({"error": "mediaSources must be a list"}), 400
    
    import time
    start_time = time.time()
    results = []
    success_count = 0
    failure_count = 0
    
    for media_source in media_sources:
        try:
            # Process each media source
            result = tagger.process_media(media_source, analysis_options)
            results.append({
                "mediaId": result["mediaId"],
                "status": "success",
                "tags": result
            })
            success_count += 1
        except Exception as e:
            results.append({
                "mediaId": media_source,
                "status": "error",
                "error": str(e)
            })
            failure_count += 1
    
    return jsonify({
        "results": results,
        "metadata": {
            "totalProcessed": len(media_sources),
            "successCount": success_count,
            "failureCount": failure_count,
            "processingTime": time.time() - start_time
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)