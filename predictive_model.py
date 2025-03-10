import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score

class DonorUpgradePrediction:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, donor_data):
        """
        Create features from donor data for prediction
        """
        features = pd.DataFrame()
        
        # Group by donor
        donor_stats = donor_data.groupby('donor_id').agg({
            'donation_amount': ['mean', 'std', 'count', 'sum'],
            'donation_date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        donor_stats.columns = [
            'donor_id',
            'avg_donation',
            'std_donation',
            'donation_count',
            'total_donated',
            'first_donation',
            'last_donation'
        ]
        
        # Calculate features
        features['donor_id'] = donor_stats['donor_id']
        features['avg_donation'] = donor_stats['avg_donation']
        features['donation_consistency'] = 1 - (
            donor_stats['std_donation'] / donor_stats['avg_donation']
        ).fillna(0).clip(0, 1)
        features['months_active'] = (
            (pd.to_datetime(donor_stats['last_donation']) -
             pd.to_datetime(donor_stats['first_donation'])).dt.days / 30
        ).round()
        features['donation_frequency'] = donor_stats['donation_count'] / \
            features['months_active'].clip(1)
        features['total_donated'] = donor_stats['total_donated']
        
        return features
    
    def train(self, features, labels):
        """
        Train the prediction model
        """
        X = self.scaler.fit_transform(features.drop('donor_id', axis=1))
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = self.model.predict(X_test)
        metrics = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred)
        }
        
        return metrics
    
    def predict(self, features):
        """
        Make predictions for new donors
        """
        X = self.scaler.transform(features.drop('donor_id', axis=1))
        probabilities = self.model.predict_proba(X)
        
        return pd.DataFrame({
            'donor_id': features['donor_id'],
            'upgrade_probability': probabilities[:, 1]
        })
