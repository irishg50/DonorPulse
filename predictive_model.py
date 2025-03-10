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
            random_state=42,
            class_weight='balanced'  # Handle imbalanced classes
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
        })

        # Flatten column names
        donor_stats.columns = [
            f"{col[0]}_{col[1]}" for col in donor_stats.columns
        ]
        donor_stats = donor_stats.reset_index()

        # Calculate features
        features['donor_id'] = donor_stats['donor_id']
        features['avg_donation'] = donor_stats['donation_amount_mean']
        features['donation_consistency'] = 1 - (
            donor_stats['donation_amount_std'] / donor_stats['donation_amount_mean']
        ).fillna(0).clip(0, 1)
        features['months_active'] = (
            (pd.to_datetime(donor_stats['donation_date_max']) -
             pd.to_datetime(donor_stats['donation_date_min'])).dt.days / 30
        ).round()
        features['donation_frequency'] = donor_stats['donation_amount_count'] / \
            features['months_active'].clip(1)
        features['total_donated'] = donor_stats['donation_amount_sum']

        return features

    def train(self, features, labels):
        """
        Train the prediction model
        """
        try:
            # Ensure we have both classes represented
            if len(np.unique(labels)) < 2:
                # If all labels are the same, create a dummy negative class
                features = pd.concat([features, features.iloc[0:1]])
                labels = np.append(labels, 1 - labels[0])

            X = self.scaler.fit_transform(features.drop('donor_id', axis=1))
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.2, random_state=42, stratify=labels
            )

            self.model.fit(X_train, y_train)

            # Calculate metrics
            y_pred = self.model.predict(X_test)
            metrics = {
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0.5
            }

            return metrics

        except Exception as e:
            print(f"Error during model training: {str(e)}")
            return {'precision': 0, 'recall': 0, 'roc_auc': 0.5}

    def predict(self, features):
        """
        Make predictions for new donors
        """
        try:
            X = self.scaler.transform(features.drop('donor_id', axis=1))
            probabilities = self.model.predict_proba(X)

            return pd.DataFrame({
                'donor_id': features['donor_id'],
                'upgrade_probability': probabilities[:, 1]
            })

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return pd.DataFrame({
                'donor_id': features['donor_id'],
                'upgrade_probability': 0.5  # Default neutral probability
            })