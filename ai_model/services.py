import pandas as pd
from io import StringIO
from .predictor import JobListingPredictor

def predict_job_listings(csv_data):
    """
    Process job listings CSV data and return predictions
    
    Args:
        csv_data: CSV string or file-like object containing job listings data
        
    Returns:
        String containing CSV data with predictions
    """
    # Read CSV data
    df = pd.read_csv(StringIO(csv_data) if isinstance(csv_data, str) else csv_data)
    
    # Get predictions
    predictor = JobListingPredictor()
    df_with_predictions = predictor.predict_from_dataframe(df)
    
    # Convert back to CSV
    output = StringIO()
    df_with_predictions.to_csv(output, index=False)
    return output.getvalue()