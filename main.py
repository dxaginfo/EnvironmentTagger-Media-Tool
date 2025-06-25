"""
EnvironmentTagger - Media Automation Tool

Core implementation for automatic environment tagging in media assets
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

import numpy as np
import tensorflow as tf
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.cloud.storage as storage
from google.cloud import firestore
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("environment-tagger")

# Initialize FastAPI app
app = FastAPI(
    title="EnvironmentTagger API",
    description="API for tagging and categorizing environmental elements in media scenes",
    version="0.6.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Cloud clients
try:
    storage_client = storage.Client()
    firestore_client = firestore.Client()
    
    # Configure Gemini API
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-pro-vision')
    
    logger.info("Successfully initialized Google Cloud clients")
except Exception as e:
    logger.error(f"Error initializing Google Cloud clients: {e}")
    # Continue without failing to allow local development
    storage_client = None
    firestore_client = None
    gemini_model = None

# Load TensorFlow model
try:
    # Using TF Hub for a pre-trained model
    model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
    base_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.hub.KerasLayer(model_url, trainable=False)
    ])
    logger.info("Successfully loaded TensorFlow model")
except Exception as e:
    logger.error(f"Error loading TensorFlow model: {e}")
    base_model = None

# Pydantic models for request/response validation
class MediaInput(BaseModel):
    type: str = Field(..., description="URL, Base64, or Cloud Storage path")
    format: str = Field(..., description="'image', 'video', or 'sequence'")
    duration: Optional[float] = Field(None, description="Duration in seconds (for video)")
    timestamp: Optional[str] = Field(None, description="ISO 8601 timestamp")

class AnalysisOptions(BaseModel):
    detectionLevel: str = Field("detailed", description="'basic', 'detailed', or 'comprehensive'")
    tagCategories: Optional[List[str]] = Field(None, description="Categories to detect")
    confidenceThreshold: float = Field(0.7, description="Confidence threshold (0.0-1.0)")
    includeFrameAnalysis: bool = Field(False, description="Include frame-by-frame analysis")
    outputFormat: str = Field("json", description="'json', 'xml', 'csv', or 'firestore'")

class ContextInfo(BaseModel):
    projectId: Optional[str] = None
    sceneId: Optional[str] = None
    previousTags: Optional[List[Dict[str, Any]]] = None

class EnvironmentTagRequest(BaseModel):
    media: MediaInput
    options: Optional[AnalysisOptions] = AnalysisOptions()
    context: Optional[ContextInfo] = ContextInfo()

class EnvironmentTag(BaseModel):
    category: str
    tag: str
    confidence: float
    timeRanges: Optional[List[Dict[str, float]]] = None
    metadata: Optional[Dict[str, Any]] = None

class SceneAnalysis(BaseModel):
    dominantEnvironment: str
    environmentChanges: Optional[List[Dict[str, Any]]] = None
    consistencyScore: float

class FrameAnalysis(BaseModel):
    frameNumber: int
    timestamp: float
    tags: List[Dict[str, Any]]

class GeminiInsights(BaseModel):
    environmentDescription: str
    continuityNotes: Optional[str] = None
    enhancementSuggestions: Optional[List[str]] = None

class MediaInfo(BaseModel):
    mediaId: str
    duration: Optional[float] = None
    frameCount: Optional[int] = None
    resolution: Optional[str] = None

class EnvironmentTagResponse(BaseModel):
    mediaInfo: MediaInfo
    environmentTags: List[EnvironmentTag]
    sceneAnalysis: SceneAnalysis
    frameAnalysis: Optional[List[FrameAnalysis]] = None
    geminiInsights: Optional[GeminiInsights] = None

# Environmental category map (simplified for example)
ENVIRONMENT_CATEGORIES = {
    "location": [
        "indoor", "outdoor", "urban", "rural", "coastal", "mountain", 
        "forest", "desert", "residential", "commercial", "industrial"
    ],
    "time": [
        "day", "night", "dawn", "dusk", "morning", "afternoon", 
        "evening", "golden-hour", "blue-hour"
    ],
    "weather": [
        "clear", "cloudy", "rainy", "snowy", "foggy", "sunny", 
        "overcast", "stormy", "windy"
    ],
    "lighting": [
        "natural", "artificial", "mixed", "bright", "dim", "high-key", 
        "low-key", "side-lit", "back-lit", "top-lit"
    ],
    "season": [
        "spring", "summer", "autumn", "winter"
    ]
}

# Utility functions
def download_from_url(url: str) -> np.ndarray:
    """Download image from URL and convert to numpy array."""
    # This is a placeholder - real implementation would handle various URL types
    # and authentication methods
    try:
        import urllib.request
        from io import BytesIO
        resp = urllib.request.urlopen(url)
        img_data = resp.read()
        img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        logger.error(f"Error downloading from URL: {e}")
        raise HTTPException(status_code=400, detail=f"Could not download from URL: {str(e)}")

def download_from_gcs(gcs_path: str) -> np.ndarray:
    """Download image from Google Cloud Storage and convert to numpy array."""
    try:
        if not storage_client:
            raise Exception("Storage client not initialized")
        
        # Parse bucket and blob names
        if gcs_path.startswith('gs://'):
            gcs_path = gcs_path[5:]
        
        bucket_name, blob_name = gcs_path.split('/', 1)
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Download to memory
        img_data = blob.download_as_bytes()
        img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        logger.error(f"Error downloading from GCS: {e}")
        raise HTTPException(status_code=400, detail=f"Could not download from GCS: {str(e)}")

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for model inference."""
    img = cv2.resize(image, (224, 224))
    img = img / 255.0  # Normalize
    return img

def extract_frames(video_path: str, num_frames: int = 10) -> List[np.ndarray]:
    """Extract frames from video for analysis."""
    try:
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            raise ValueError("No frames found in video")
            
        # Select evenly spaced frames
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
        cap.release()
        return frames
    except Exception as e:
        logger.error(f"Error extracting frames: {e}")
        raise HTTPException(status_code=400, detail=f"Could not extract video frames: {str(e)}")

def classify_environment(image: np.ndarray) -> Dict[str, List[Dict[str, float]]]:
    """Classify environment types in image."""
    # This is a simplified implementation
    # A real implementation would use a more sophisticated model
    
    if base_model is None:
        # Return mock data if model not loaded
        return {
            "location": [
                {"tag": "indoor", "confidence": 0.85},
                {"tag": "residential", "confidence": 0.76}
            ],
            "time": [
                {"tag": "day", "confidence": 0.92}
            ],
            "lighting": [
                {"tag": "natural", "confidence": 0.88}
            ]
        }
    
    try:
        # Preprocess image
        processed_img = preprocess_image(image)
        processed_img = np.expand_dims(processed_img, axis=0)
        
        # Get model predictions
        predictions = base_model.predict(processed_img)
        
        # For this example, we'll generate synthetic environment tags
        # A real implementation would map model outputs to environment categories
        
        # Convert predictions to environment tags (simplified)
        top_predictions = tf.nn.top_k(predictions[0], k=5)
        
        # Mock environment classification based on image characteristics
        brightness = np.mean(image)
        color_std = np.std(image, axis=(0,1))
        color_variance = np.sum(color_std)
        
        results = {}
        
        # Location detection based on image features
        if brightness > 150:
            results["location"] = [
                {"tag": "outdoor", "confidence": 0.8 + (brightness-150)/255},
                {"tag": "urban" if color_variance > 50 else "rural", "confidence": 0.75}
            ]
        else:
            results["location"] = [
                {"tag": "indoor", "confidence": 0.8 + (150-brightness)/150},
                {"tag": "residential", "confidence": 0.7}
            ]
        
        # Time detection
        if brightness > 180:
            results["time"] = [{"tag": "day", "confidence": 0.9}]
        elif brightness > 120:
            results["time"] = [{"tag": "evening", "confidence": 0.8}]
        else:
            results["time"] = [{"tag": "night", "confidence": 0.85}]
            
        # Lighting detection
        if color_variance > 60:
            results["lighting"] = [{"tag": "high-contrast", "confidence": 0.75}]
        else:
            results["lighting"] = [{"tag": "even", "confidence": 0.8}]
            
        # Filter by confidence threshold
        for category, tags in results.items():
            results[category] = [tag for tag in tags if tag["confidence"] > 0.7]
            
        return results
    except Exception as e:
        logger.error(f"Error classifying environment: {e}")
        # Return fallback classification
        return {
            "location": [{"tag": "unknown", "confidence": 0.5}],
            "time": [{"tag": "unknown", "confidence": 0.5}]
        }

async def analyze_with_gemini(image: np.ndarray) -> Dict[str, Any]:
    """Analyze image with Gemini API for enhanced understanding."""
    if not gemini_model:
        # Return mock data if Gemini not configured
        return {
            "environmentDescription": "Indoor residential setting with natural lighting from windows.",
            "continuityNotes": "Consistent environment suitable for dialogue scenes.",
            "enhancementSuggestions": ["Consider warmer color temperature for more inviting atmosphere."]
        }
    
    try:
        # Convert numpy array to bytes for Gemini
        success, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Failed to encode image")
            
        image_bytes = buffer.tobytes()
        
        # Create the prompt for Gemini
        prompt = """
        Analyze this image and provide detailed information about the environment:
        1. Describe the environment in detail (location, time of day, weather, lighting, season)
        2. Identify any notable environmental elements
        3. Note any potential continuity considerations for filmmaking
        4. Suggest possible enhancements for this environment
        
        Format your response as JSON with the following structure:
        {
          "environmentDescription": "detailed description",
          "environmentElements": ["element1", "element2", ...],
          "continuityNotes": "notes about continuity",
          "enhancementSuggestions": ["suggestion1", "suggestion2", ...]
        }
        """
        
        # Call Gemini API
        response = gemini_model.generate_content([prompt, image_bytes])
        
        # Extract and parse JSON from response
        response_text = response.text
        
        # Find JSON in the response
        import re
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text
            
        # Clean up and parse JSON
        try:
            # Remove markdown code blocks if present
            if json_str.startswith('```') and json_str.endswith('```'):
                json_str = json_str[3:-3]
                
            result = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Couldn't parse Gemini response as JSON")
            # Extract information with regex as fallback
            env_desc_match = re.search(r'"environmentDescription":\s*"([^"]+)"', json_str)
            env_desc = env_desc_match.group(1) if env_desc_match else "Unknown environment"
            
            cont_notes_match = re.search(r'"continuityNotes":\s*"([^"]+)"', json_str)
            cont_notes = cont_notes_match.group(1) if cont_notes_match else "No continuity notes available"
            
            result = {
                "environmentDescription": env_desc,
                "continuityNotes": cont_notes,
                "enhancementSuggestions": ["Improve lighting for better visibility"]
            }
            
        return result
    except Exception as e:
        logger.error(f"Error with Gemini analysis: {e}")
        return {
            "environmentDescription": "Environment analysis unavailable",
            "continuityNotes": "Unable to process continuity information",
            "enhancementSuggestions": ["Try resubmitting with a clearer image"]
        }

def save_to_firestore(analysis_result: Dict[str, Any], context: ContextInfo) -> str:
    """Save analysis results to Firestore."""
    if not firestore_client:
        return "mock-document-id-12345"  # Mock ID for local development
        
    try:
        # Create document reference
        collection = "environmentTags"
        if context and context.projectId:
            collection = f"projects/{context.projectId}/environmentTags"
            
        doc_ref = firestore_client.collection(collection).document()
        
        # Add metadata
        analysis_result["metadata"] = {
            "timestamp": firestore.SERVER_TIMESTAMP,
            "version": "0.6.0"
        }
        
        if context and context.sceneId:
            analysis_result["metadata"]["sceneId"] = context.sceneId
            
        # Save to Firestore
        doc_ref.set(analysis_result)
        
        return doc_ref.id
    except Exception as e:
        logger.error(f"Error saving to Firestore: {e}")
        return f"error-{datetime.now().timestamp()}"

# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "EnvironmentTagger API",
        "version": "0.6.0",
        "status": "operational"
    }

@app.post("/analyze", response_model=EnvironmentTagResponse)
async def analyze_environment(
    request: EnvironmentTagRequest,
    authorization: Optional[str] = Header(None)
):
    """Analyze media for environmental elements."""
    # In a production environment, verify authorization
    if not authorization and os.environ.get("REQUIRE_AUTH", "false").lower() == "true":
        raise HTTPException(status_code=401, detail="Authorization required")
        
    try:
        # Process based on media type
        media = request.media
        media_id = f"media-{datetime.now().timestamp()}"
        
        # Handle different media sources
        if media.type.startswith("http"):
            # URL media
            if media.format == "image":
                image = download_from_url(media.type)
                frames = [image]
            elif media.format == "video":
                # For this example, we'd download the video and extract frames
                # This is simplified - would need proper video handling
                frames = [download_from_url(media.type)]  # Placeholder
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {media.format}")
                
        elif media.type.startswith("gs://"):
            # Google Cloud Storage
            if media.format == "image":
                image = download_from_gcs(media.type)
                frames = [image]
            elif media.format == "video":
                # Simplified - would need proper video handling
                frames = [download_from_gcs(media.type)]  # Placeholder
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {media.format}")
        else:
            # Assume direct base64 or other format
            raise HTTPException(status_code=400, detail="Unsupported media type")
            
        # Media info
        media_info = MediaInfo(
            mediaId=media_id,
            resolution=f"{frames[0].shape[1]}x{frames[0].shape[0]}",
        )
        
        if media.format == "video" and media.duration:
            media_info.duration = media.duration
            media_info.frameCount = len(frames)
            
        # Analyze frames
        all_tags = []
        frame_analyses = []
        
        for i, frame in enumerate(frames):
            # Basic environment classification
            env_tags = classify_environment(frame)
            
            # Convert to EnvironmentTag format
            for category, tags in env_tags.items():
                for tag_info in tags:
                    all_tags.append(
                        EnvironmentTag(
                            category=category,
                            tag=tag_info["tag"],
                            confidence=tag_info["confidence"],
                            timeRanges=[{"start": 0, "end": media.duration or 0}] if media.format == "video" else None
                        )
                    )
            
            # Add frame analysis if requested
            if request.options and request.options.includeFrameAnalysis:
                frame_analyses.append(
                    FrameAnalysis(
                        frameNumber=i,
                        timestamp=i * (media.duration / len(frames)) if media.duration else 0,
                        tags=[{"category": cat, "tag": tag["tag"], "confidence": tag["confidence"]} 
                               for cat, tags_list in env_tags.items() 
                               for tag in tags_list]
                    )
                )
        
        # Analyze first frame with Gemini for enhanced insights
        gemini_results = await analyze_with_gemini(frames[0])
        
        # Create scene analysis
        # In a real implementation, this would be more sophisticated
        dominant_env = "indoor"
        if any(tag.tag == "outdoor" and tag.confidence > 0.8 for tag in all_tags if tag.category == "location"):
            dominant_env = "outdoor"
            
        scene_analysis = SceneAnalysis(
            dominantEnvironment=dominant_env,
            environmentChanges=[],  # Would detect changes across frames in video
            consistencyScore=0.95  # Placeholder - would calculate from frame consistency
        )
        
        # Create Gemini insights
        gemini_insights = GeminiInsights(
            environmentDescription=gemini_results.get("environmentDescription", "No description available"),
            continuityNotes=gemini_results.get("continuityNotes", None),
            enhancementSuggestions=gemini_results.get("enhancementSuggestions", None)
        )
        
        # Create response
        response = EnvironmentTagResponse(
            mediaInfo=media_info,
            environmentTags=all_tags,
            sceneAnalysis=scene_analysis,
            frameAnalysis=frame_analyses if request.options and request.options.includeFrameAnalysis else None,
            geminiInsights=gemini_insights
        )
        
        # Save to Firestore if requested
        if request.options and request.options.outputFormat == "firestore":
            doc_id = save_to_firestore(response.dict(), request.context)
            response.mediaInfo.mediaId = doc_id
            
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing media: {str(e)}")

@app.get("/categories")
async def get_categories():
    """Get available environment categories and tags."""
    return ENVIRONMENT_CATEGORIES

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.6.0",
        "services": {
            "tensorflow": "available" if base_model is not None else "unavailable",
            "gemini": "available" if gemini_model is not None else "unavailable",
            "storage": "available" if storage_client is not None else "unavailable",
            "firestore": "available" if firestore_client is not None else "unavailable"
        }
    }

# For Cloud Functions deployment
def analyze_environment_http(request):
    """Cloud Function entry point for HTTP trigger."""
    import functions_framework
    from flask import Request, Response
    
    # Convert Flask request to FastAPI compatible
    body = request.get_json(silent=True)
    if not body:
        return Response(
            response=json.dumps({"error": "Invalid request body"}),
            status=400,
            mimetype="application/json"
        )
    
    # Create FastAPI request
    try:
        env_request = EnvironmentTagRequest(**body)
        result = analyze_environment(env_request, request.headers.get("Authorization"))
        return Response(
            response=json.dumps(result),
            status=200,
            mimetype="application/json"
        )
    except Exception as e:
        return Response(
            response=json.dumps({"error": str(e)}),
            status=500,
            mimetype="application/json"
        )

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)