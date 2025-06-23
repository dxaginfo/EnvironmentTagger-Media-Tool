import os
import tempfile
import urllib.request
from flask import Flask, request, jsonify
from environment_detector import EnvironmentDetector

app = Flask(__name__)

# Initialize environment detector with environment variables
PROJECT_ID = os.environ.get('PROJECT_ID', 'your-project-id')
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'environment-tags')

detector = EnvironmentDetector(PROJECT_ID, BUCKET_NAME)

@app.route('/api/tag', methods=['POST'])
def tag_media():
    """API endpoint to tag media with environment metadata"""
    data = request.json
    media_url = data.get('mediaUrl')
    options = data.get('options', {})
    
    if not media_url:
        return jsonify({'error': 'Media URL is required'}), 400
    
    try:
        # Download media file
        local_path = download_media(media_url)
        
        # Process media file
        if is_video(local_path):
            frames = extract_frames(local_path)
            results = []
            for frame in frames:
                result = detector.detect_environment(frame)
                results.append(result)
            # Aggregate results
            environment = aggregate_environment_results(results)
        else:
            environment = detector.detect_environment(local_path)
        
        # Store results
        media_id = detector.store_environment_metadata(environment)
        
        # Clean up
        os.remove(local_path)
        
        return jsonify({
            'mediaId': media_id,
            'environment': environment
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tags/<media_id>', methods=['GET'])
def get_tags(media_id):
    """API endpoint to retrieve environment tags for a specific media"""
    try:
        environment = detector.get_environment_metadata(media_id)
        
        if not environment:
            return jsonify({'error': 'Media ID not found'}), 404
            
        return jsonify(environment)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tags/<media_id>', methods=['PUT'])
def update_tags(media_id):
    """API endpoint to update environment tags for a specific media"""
    data = request.json
    tags = data.get('tags', [])
    override = data.get('override', False)
    
    if not tags:
        return jsonify({'error': 'Tags are required'}), 400
    
    try:
        success = detector.update_environment_metadata(media_id, tags, override)
        
        if not success:
            return jsonify({'error': 'Failed to update tags'}), 404
            
        # Get updated metadata
        environment = detector.get_environment_metadata(media_id)
        return jsonify(environment)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def download_media(url):
    """Download media from URL to local temp file"""
    _, temp_path = tempfile.mkstemp()
    urllib.request.urlretrieve(url, temp_path)
    return temp_path

def is_video(file_path):
    """Check if file is video based on extension"""
    video_extensions = ['.mp4', '.mov', '.avi', '.wmv', '.mkv']
    _, ext = os.path.splitext(file_path.lower())
    return ext in video_extensions

def extract_frames(video_path):
    """Extract representative frames from video using OpenCV"""
    # This would use OpenCV to extract frames
    # For now, we'll just return the video path as if it were a single frame
    return [video_path]

def aggregate_environment_results(results):
    """Combine results from multiple frames"""
    # Simple aggregation - take the first result for now
    # A real implementation would do more sophisticated aggregation
    if results:
        return results[0]
    return {}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))