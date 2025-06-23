import os
import json
import time
import logging
from typing import Dict, List, Any, Optional

from google.cloud import vision
from google.cloud import storage
from google.cloud.firestore import Client as FirestoreClient
from google.generativeai import GenerativeModel
from PIL import Image
import io
import numpy as np
import tensorflow as tf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("environment_tagger")

class EnvironmentTagger:
    """Main class for detecting and tagging environmental elements in media."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the EnvironmentTagger with configuration."""
        self.config = config or {}
        self.detection_threshold = self.config.get("detection_threshold", 0.7)
        self.max_tags = self.config.get("max_tags", 25)
        self.gemini_model = GenerativeModel("gemini-pro-vision")
        self.vision_client = vision.ImageAnnotatorClient()
        self.storage_client = storage.Client()
        self.firestore_client = FirestoreClient()
        
        # Load TensorFlow model if specified
        tf_model_path = self.config.get("tf_model_path")
        if tf_model_path and os.path.exists(tf_model_path):
            self.tf_model = tf.saved_model.load(tf_model_path)
        else:
            self.tf_model = None
            
        # Load custom tag dictionary if specified
        custom_tag_dict_path = self.config.get("custom_tag_dictionary")
        if custom_tag_dict_path and os.path.exists(custom_tag_dict_path):
            with open(custom_tag_dict_path, 'r') as f:
                self.custom_tags = json.load(f)
        else:
            self.custom_tags = None
    
    def process_media(self, media_source: str, analysis_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a media file and generate environment tags.
        
        Args:
            media_source: URL, Cloud Storage path, or local file path
            analysis_options: Optional configuration overrides
            
        Returns:
            Dictionary containing environment analysis results
        """
        start_time = time.time()
        options = analysis_options or {}
        
        # Override config with request options if provided
        detection_threshold = options.get("detectionThreshold", self.detection_threshold)
        max_tags = options.get("maxTags", self.max_tags)
        include_confidence = options.get("includeConfidenceScores", True)
        
        # Load the media content
        image_bytes = self._load_media(media_source)
        
        # Perform environment analysis
        vision_results = self._analyze_with_vision_api(image_bytes)
        gemini_results = self._analyze_with_gemini(image_bytes)
        
        # Combine results and filter by threshold
        tags = self._combine_and_filter_results(vision_results, gemini_results, detection_threshold)
        
        # Limit to max_tags
        tags = tags[:max_tags]
        
        # Generate environment summary
        summary = self._generate_environment_summary(tags, gemini_results.get("environment_summary", ""))
        
        # Extract dominant settings
        dominant_settings = self._extract_dominant_settings(tags, vision_results)
        
        # Format final response
        response = {
            "mediaId": self._generate_media_id(media_source),
            "analysisTimestamp": self._get_timestamp(),
            "environmentTags": tags if include_confidence else [
                {k: v for k, v in tag.items() if k != "confidence"}
                for tag in tags
            ],
            "environmentSummary": summary,
            "dominantSettings": dominant_settings,
            "metadata": {
                "processingTime": time.time() - start_time,
                "modelVersion": self.config.get("model_version", "v1.0.0"),
                "additionalInfo": {}
            }
        }
        
        # Store results if requested
        if options.get("store_results", False):
            self._store_results(response)
            
        return response
    
    def _load_media(self, media_source: str) -> bytes:
        """Load media content from the provided source.
        
        Args:
            media_source: URL, Cloud Storage path, or local file path
            
        Returns:
            Binary content of the media file
        """
        if media_source.startswith("gs://"):
            # Google Cloud Storage path
            bucket_name, blob_path = media_source[5:].split("/", 1)
            bucket = self.storage_client.get_bucket(bucket_name)
            blob = bucket.blob(blob_path)
            return blob.download_as_bytes()
        elif media_source.startswith(("http://", "https://")):
            # URL - use requests library
            import requests
            response = requests.get(media_source)
            response.raise_for_status()
            return response.content
        elif os.path.exists(media_source):
            # Local file path
            with open(media_source, "rb") as f:
                return f.read()
        else:
            raise ValueError(f"Unable to load media from source: {media_source}")
    
    def _analyze_with_vision_api(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyze the image using Google Cloud Vision API.
        
        Args:
            image_bytes: Binary image data
            
        Returns:
            Dictionary with Vision API analysis results
        """
        image = vision.Image(content=image_bytes)
        
        # Request multiple feature types
        features = [
            vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION),
            vision.Feature(type_=vision.Feature.Type.LANDMARK_DETECTION),
            vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES),
            vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION),
        ]
        
        response = self.vision_client.annotate_image({
            'image': image,
            'features': features,
        })
        
        # Format response into a standardized structure
        return {
            'labels': [
                {
                    'description': label.description,
                    'score': label.score,
                    'category': self._categorize_label(label.description)
                } 
                for label in response.label_annotations
            ],
            'landmarks': [
                {
                    'description': landmark.description,
                    'score': landmark.score,
                    'locations': [
                        {
                            'latitude': location.latitude,
                            'longitude': location.longitude
                        }
                        for location in landmark.locations
                    ]
                }
                for landmark in response.landmark_annotations
            ],
            'colors': [
                {
                    'color': {
                        'red': color.color.red,
                        'green': color.color.green,
                        'blue': color.color.blue
                    },
                    'score': color.score,
                    'pixel_fraction': color.pixel_fraction
                }
                for color in response.image_properties_annotation.dominant_colors.colors
            ],
            'objects': [
                {
                    'name': obj.name,
                    'score': obj.score,
                    'bounding_box': {
                        'x1': obj.bounding_poly.normalized_vertices[0].x,
                        'y1': obj.bounding_poly.normalized_vertices[0].y,
                        'x2': obj.bounding_poly.normalized_vertices[2].x,
                        'y2': obj.bounding_poly.normalized_vertices[2].y
                    }
                }
                for obj in response.localized_object_annotations
            ]
        }
    
    def _analyze_with_gemini(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyze the image using Gemini Pro Vision API.
        
        Args:
            image_bytes: Binary image data
            
        Returns:
            Dictionary with Gemini analysis results
        """
        prompt_template = """
        Task: {task}
        
        Please analyze the environment in this image thoroughly. 
        
        Output format: {output_format}
        
        Provide your analysis in JSON format with the following structure:
        {{
          "environment_tags": [
            {{
              "tag": "[tag name]",
              "category": "[location|lighting|weather|time-of-day|mood]",
              "confidence": [0.0-1.0 value]
            }}
          ],
          "environment_summary": "[brief natural language description of the environment]"
        }}
        """
        
        prompt = prompt_template.format(
            task="Analyze and tag the environment in this image",
            output_format="Provide tags for: location type, lighting conditions, weather, time of day, mood/atmosphere"
        )
        
        response = self.gemini_model.generate_content([prompt, image_bytes])
        
        # Parse the response - handling potential errors in response format
        try:
            # Find JSON in the response
            response_text = response.text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start >= 0 and json_end > json_start:
                json_string = response_text[json_start:json_end+1]
                return json.loads(json_string)
            else:
                # Structured parsing failed, create manual structure
                return {
                    "environment_tags": self._extract_tags_from_text(response_text),
                    "environment_summary": self._extract_summary_from_text(response_text)
                }
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            return {
                "environment_tags": [],
                "environment_summary": ""
            }
    
    def _categorize_label(self, label: str) -> str:
        """Categorize a label into an environmental category.
        
        Args:
            label: The label text to categorize
            
        Returns:
            Category string (location, lighting, weather, time-of-day, or general)
        """
        # Location-related terms
        location_terms = ['indoor', 'outdoor', 'urban', 'rural', 'forest', 'beach', 
                         'mountain', 'office', 'home', 'studio', 'street', 'room', 
                         'building', 'park', 'landscape', 'city', 'wilderness',
                         'desert', 'ocean', 'lake', 'river', 'restaurant', 'cafe']
        
        # Lighting-related terms
        lighting_terms = ['bright', 'dark', 'shadow', 'sunlight', 'daylight', 
                         'artificial light', 'natural light', 'backlight', 'spotlight',
                         'flash', 'dim', 'fluorescent', 'incandescent', 'led', 
                         'ambient', 'harsh', 'soft', 'fill light', 'key light']
        
        # Weather-related terms
        weather_terms = ['sunny', 'cloudy', 'rainy', 'snowy', 'foggy', 'misty',
                        'stormy', 'windy', 'clear', 'overcast', 'hazy', 'humid',
                        'dry', 'hot', 'cold', 'warm', 'cool', 'frost', 'icy']
        
        # Time-of-day related terms
        time_terms = ['day', 'night', 'morning', 'evening', 'dusk', 'dawn',
                     'afternoon', 'sunrise', 'sunset', 'twilight', 'noon',
                     'midnight', 'golden hour', 'blue hour']
        
        # Check custom category mappings if available
        if self.custom_tags and label in self.custom_tags:
            return self.custom_tags.get(label, {}).get('category', 'general')
        
        # Check against standard categories
        label_lower = label.lower()
        
        for term in location_terms:
            if term in label_lower:
                return 'location'
                
        for term in lighting_terms:
            if term in label_lower:
                return 'lighting'
                
        for term in weather_terms:
            if term in label_lower:
                return 'weather'
                
        for term in time_terms:
            if term in label_lower:
                return 'time-of-day'
        
        # Default category if no match is found
        return 'general'
    
    def _combine_and_filter_results(self, vision_results: Dict[str, Any], 
                                  gemini_results: Dict[str, Any],
                                  threshold: float) -> List[Dict[str, Any]]:
        """Combine and filter results from different analysis sources.
        
        Args:
            vision_results: Results from Vision API
            gemini_results: Results from Gemini API
            threshold: Minimum confidence threshold for tags
            
        Returns:
            List of filtered and combined environment tags
        """
        combined_tags = []
        
        # Add Vision API labels
        for label in vision_results.get('labels', []):
            if label['score'] >= threshold:
                combined_tags.append({
                    'tag': label['description'],
                    'category': label['category'],
                    'confidence': float(label['score']),
                    'source': 'vision_api'
                })
        
        # Add landmark information as location tags
        for landmark in vision_results.get('landmarks', []):
            if landmark['score'] >= threshold:
                combined_tags.append({
                    'tag': landmark['description'],
                    'category': 'location',
                    'confidence': float(landmark['score']),
                    'source': 'vision_api',
                    'coordinates': landmark.get('locations', [])[0] if landmark.get('locations') else None
                })
        
        # Add Gemini tags
        for tag in gemini_results.get('environment_tags', []):
            # Skip tags with low confidence
            if tag.get('confidence', 0) < threshold:
                continue
                
            # Check if tag already exists from Vision API
            existing_tag = next((t for t in combined_tags if t['tag'].lower() == tag['tag'].lower()), None)
            
            if existing_tag:
                # Update with higher confidence if Gemini is more confident
                if tag.get('confidence', 0) > existing_tag.get('confidence', 0):
                    existing_tag['confidence'] = tag.get('confidence', 0)
                    existing_tag['source'] = 'gemini'
            else:
                # Add new tag from Gemini
                combined_tags.append({
                    'tag': tag['tag'],
                    'category': tag.get('category', 'general'),
                    'confidence': tag.get('confidence', 0.7),
                    'source': 'gemini'
                })
        
        # Sort by confidence (descending)
        combined_tags.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Add bounding boxes for environmental objects when available
        for tag in combined_tags:
            if tag['source'] == 'vision_api':
                # Find matching object with bounding box
                matching_obj = next((obj for obj in vision_results.get('objects', []) 
                                   if obj['name'].lower() == tag['tag'].lower()), None)
                
                if matching_obj and 'bounding_box' in matching_obj:
                    tag['boundingBox'] = matching_obj['bounding_box']
        
        return combined_tags
    
    def _generate_environment_summary(self, tags: List[Dict[str, Any]], 
                                    gemini_summary: str) -> str:
        """Generate a natural language summary of the environment.
        
        Args:
            tags: List of environment tags
            gemini_summary: Summary from Gemini if available
            
        Returns:
            Natural language summary of the environment
        """
        # If Gemini provided a good summary, use it
        if gemini_summary and len(gemini_summary) > 20:
            return gemini_summary
        
        # Otherwise, generate a summary from the tags
        location_tags = [tag['tag'] for tag in tags if tag.get('category') == 'location']
        lighting_tags = [tag['tag'] for tag in tags if tag.get('category') == 'lighting']
        weather_tags = [tag['tag'] for tag in tags if tag.get('category') == 'weather']
        time_tags = [tag['tag'] for tag in tags if tag.get('category') == 'time-of-day']
        
        summary_parts = []
        
        if location_tags:
            summary_parts.append(f"This is a {', '.join(location_tags[:2])} environment")
            
        if lighting_tags:
            summary_parts.append(f"with {', '.join(lighting_tags[:2])} lighting")
            
        if weather_tags:
            summary_parts.append(f"during {', '.join(weather_tags[:2])} weather conditions")
            
        if time_tags:
            summary_parts.append(f"during {', '.join(time_tags[:1])}")
            
        if not summary_parts:
            # Fallback to general tags if no categorized tags available
            general_tags = [tag['tag'] for tag in tags[:3]]
            summary = f"Environment containing {', '.join(general_tags)}"
        else:
            summary = ". ".join(summary_parts) + "."
            
        return summary
    
    def _extract_dominant_settings(self, tags: List[Dict[str, Any]], 
                                 vision_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract the dominant environmental settings.
        
        Args:
            tags: List of environment tags
            vision_results: Results from Vision API
            
        Returns:
            List of dominant environmental settings
        """
        dominant_settings = []
        
        # Add highest confidence tag per category
        categories = {}
        for tag in tags:
            category = tag.get('category', 'general')
            if category not in categories or tag['confidence'] > categories[category]['confidence']:
                categories[category] = tag
                
        for category, tag in categories.items():
            dominant_settings.append({
                'type': category,
                'value': tag['tag']
            })
            
        # Add dominant color information
        if vision_results.get('colors') and len(vision_results['colors']) > 0:
            top_color = vision_results['colors'][0]
            color = top_color['color']
            rgb = f"rgb({color['red']}, {color['green']}, {color['blue']})"
            
            dominant_settings.append({
                'type': 'dominant_color',
                'value': rgb
            })
            
        return dominant_settings
    
    def _extract_tags_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract environment tags from unstructured text.
        
        Args:
            text: Raw text response from Gemini
            
        Returns:
            List of structured environment tags
        """
        tags = []
        
        # Simple heuristic extraction
        category_markers = [
            ('location', ['location', 'place', 'setting', 'environment']),
            ('lighting', ['lighting', 'light']),
            ('weather', ['weather', 'atmospheric']),
            ('time-of-day', ['time', 'day', 'night', 'morning', 'evening']),
            ('mood', ['mood', 'atmosphere', 'feeling', 'ambiance'])
        ]
        
        lines = text.strip().split('\n')
        current_category = 'general'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line indicates a category
            for category, markers in category_markers:
                if any(marker.lower() in line.lower() for marker in markers):
                    current_category = category
                    break
                    
            # Extract potential tags from line
            potential_tags = [
                tag.strip() for tag in line.split(':')[-1].split(',')
                if tag.strip() and not any(m in tag.lower() for c, markers in category_markers for m in markers)
            ]
            
            for tag in potential_tags:
                # Remove any rating indicators and extract confidence if present
                confidence = 0.7  # Default confidence
                if '(' in tag and ')' in tag:
                    # Check for confidence in parentheses like "indoor (0.9)"
                    tag_parts = tag.split('(')
                    tag_name = tag_parts[0].strip()
                    conf_part = tag_parts[1].split(')')[0]
                    
                    try:
                        # Try to convert to float if it looks like a confidence value
                        if conf_part.replace('.', '', 1).isdigit():
                            confidence = float(conf_part)
                            if confidence > 1:
                                confidence = confidence / 10  # Handle 1-10 scale
                            tag = tag_name
                    except ValueError:
                        pass  # Not a confidence value, keep original tag
                
                tags.append({
                    'tag': tag,
                    'category': current_category,
                    'confidence': confidence
                })
        
        return tags
    
    def _extract_summary_from_text(self, text: str) -> str:
        """Extract environment summary from unstructured text.
        
        Args:
            text: Raw text response from Gemini
            
        Returns:
            Environment summary string
        """
        # Look for summary section
        summary_markers = ['summary', 'description', 'overview']
        lines = text.strip().split('\n')
        
        for i, line in enumerate(lines):
            if any(marker.lower() in line.lower() for marker in summary_markers):
                # Return the next non-empty line if it exists
                for j in range(i+1, len(lines)):
                    if lines[j].strip() and not any(marker.lower() in lines[j].lower() for marker in summary_markers):
                        return lines[j].strip()
        
        # If no clear summary found, take the first substantial paragraph
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            return paragraphs[0].strip()
            
        return ""
    
    def _generate_media_id(self, media_source: str) -> str:
        """Generate a unique ID for the media source.
        
        Args:
            media_source: Original media source string
            
        Returns:
            Unique ID string
        """
        import hashlib
        return hashlib.md5(media_source.encode()).hexdigest()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO 8601 format."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"
    
    def _store_results(self, results: Dict[str, Any]) -> None:
        """Store analysis results in Firestore.
        
        Args:
            results: Analysis results dictionary
        """
        try:
            collection = self.firestore_client.collection("environment_tags")
            collection.document(results["mediaId"]).set(results)
        except Exception as e:
            logger.error(f"Error storing results in Firestore: {e}")

# Cloud Function entry point
def process_media(request):
    """Cloud Function entry point for processing media.
    
    Args:
        request: HTTP request object
        
    Returns:
        HTTP response with analysis results
    """
    from flask import jsonify
    
    try:
        request_json = request.get_json(silent=True)
        
        if not request_json or 'mediaSource' not in request_json:
            return jsonify({"error": "Invalid request: mediaSource is required"}), 400
            
        media_source = request_json.get("mediaSource")
        analysis_options = request_json.get("analysisOptions", {})
        
        # Initialize tagger with default config
        tagger = EnvironmentTagger()
        
        # Process the media
        results = tagger.process_media(media_source, analysis_options)
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

# Command-line execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process media and tag environments")
    parser.add_argument("--media", required=True, help="Path to media file")
    parser.add_argument("--threshold", type=float, default=0.7, help="Detection threshold")
    parser.add_argument("--max-tags", type=int, default=25, help="Maximum number of tags")
    parser.add_argument("--output", default="json", choices=["json", "csv", "firestore"],
                        help="Output format")
    args = parser.parse_args()
    
    # Initialize tagger with command-line config
    tagger = EnvironmentTagger({
        "detection_threshold": args.threshold,
        "max_tags": args.max_tags
    })
    
    # Process the media
    results = tagger.process_media(args.media, {
        "outputFormat": args.output
    })
    
    # Output results
    if args.output == "json":
        print(json.dumps(results, indent=2))
    elif args.output == "csv":
        import csv
        import sys
        
        writer = csv.writer(sys.stdout)
        writer.writerow(["Tag", "Category", "Confidence"])
        
        for tag in results["environmentTags"]:
            writer.writerow([tag["tag"], tag["category"], tag["confidence"]])
    # Firestore output is handled by the tagger itself