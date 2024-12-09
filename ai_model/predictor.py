import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datetime import datetime
import re
from django.conf import settings
import os
from ai_model.models import JobListingClassifier, JobListingDataset


class JobListingPredictor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(JobListingPredictor, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_path = os.path.join(settings.BASE_DIR, 'ai_model', 'model.pth')
            self.model = self.load_model(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.initialized = True

    @staticmethod
    def load_model(model_path):
        model = JobListingClassifier(metadata_size=2)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        return model

    @staticmethod
    def preprocess_data(df):
        # Clean text data
        df['cleaned_description'] = df.apply(
            lambda x: f"{x['Title']} {x['JobDescription']} {x['JobRequirment']}",
            axis=1
        ).apply(lambda x: JobListingPredictor._clean_text(x))

        # Process dates
        df['posting_date'] = pd.to_datetime(df['StartDate'], errors='coerce')
        df['posting_age'] = (datetime.now() - df['posting_date']).dt.days.fillna(0)

        # Process salary
        df['avg_salary'] = df['Salary'].apply(lambda x: JobListingPredictor._extract_average_salary(x))

        # Prepare features
        text_data = df['cleaned_description'].values
        metadata = df[['posting_age', 'avg_salary']].astype(float).values
        metadata_tensor = torch.FloatTensor(metadata)

        return text_data, metadata_tensor

    @staticmethod
    def _clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def _extract_average_salary(salary_str):
        try:
            salary_str = str(salary_str)
            numbers = re.findall(r'\d+', salary_str.replace(',', ''))
            if len(numbers) >= 2:
                return float((int(numbers[0]) + int(numbers[1])) / 2)
            return float(numbers[0]) if numbers else 0.0
        except:
            return 0.0

    def predict_from_dataframe(self, df):
        """
        Predict job listing classifications for a DataFrame
        Returns original DataFrame with new 'prediction' column
        """
        self.model.eval()

        # Preprocess the data
        text_data, metadata_tensor = self.preprocess_data(df)

        # Create dataset and dataloader
        dataset = JobListingDataset(text_data, metadata_tensor)
        dataloader = DataLoader(dataset, batch_size=16)

        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                metadata = batch['metadata'].to(self.device)

                outputs = self.model(input_ids, attention_mask, metadata)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())

        # Add predictions to DataFrame
        df['prediction'] = predictions

        return df