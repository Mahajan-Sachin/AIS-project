import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

class AISIntrusionDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.scaler = StandardScaler()
        self.encoders = {}

    def preprocess(self, df):
        df = df.copy()

        # Encode categorical features
        cat_cols = ['protocol_type', 'service', 'flag']
        for col in cat_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col])
            else:
                df[col] = self.encoders[col].transform(df[col])

        # Scale features
        df_scaled = self.scaler.fit_transform(df)
        return df_scaled

    def fit(self, df):
        data = self.preprocess(df)
        self.model.fit(data)

    def predict(self, df):
        data = self.preprocess(df)
        predictions = self.model.predict(data)
        scores = self.model.decision_function(data)
        result_df = df.copy()
        result_df['anomaly'] = predictions
        result_df['risk_score'] = -scores  # Higher is worse
        result_df['alert'] = result_df['anomaly'].apply(lambda x: 'Intrusion' if x == -1 else 'Normal')
        return result_df

    def save_model(self, path):
        joblib.dump((self.model, self.scaler, self.encoders), path)

    def load_model(self, path):
        self.model, self.scaler, self.encoders = joblib.load(path)
