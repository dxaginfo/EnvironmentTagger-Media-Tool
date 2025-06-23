import functions_framework
import sys
import os

# Add parent directory to path to import environment_tagger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment_tagger import EnvironmentTagger, process_media

@functions_framework.http
def environment_tagger_http(request):
    """HTTP Cloud Function entry point.
    
    Args:
        request: Flask request object
        
    Returns:
        Response from the process_media function
    """
    return process_media(request)