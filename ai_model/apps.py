from django.apps import AppConfig
from django.conf import settings
import os

class AIModelConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ai_model'

    def ready(self):
        # Import here to avoid circular import
        from .predictor import JobListingPredictor
        
        # Check if running in main process (avoid duplicate loading in development)
        if os.environ.get('RUN_MAIN', None) != 'true':
            print("Initializing AI Model...")
            # Initialize the predictor
            JobListingPredictor()
            print("AI Model loaded successfully!")