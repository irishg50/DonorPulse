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
            class_weight='balanced'
        )
        self.scaler = StandardScaler()

    def prepare_features(self, donor_data):
        """
        Create features from donor data for prediction
        """
        features = pd.DataFrame()

        # Group by donor and calculate rolling averages
        donor_stats = []

        for donor_id in donor_data['donor_id'].unique():
            donor_history = donor_data[donor_data['donor_id'] == donor_id].sort_values('donation_date')

            # Calculate 6-month rolling average
            rolling_avg = donor_history['donation_amount'].rolling(window=6, min_periods=1).mean()

            # Calculate growth rates
            growth_rate = (donor_history['donation_amount'] - donor_history['donation_amount'].shift(6)) / \
                          donor_history['donation_amount'].shift(6)

            # Get latest stats
            latest_stats = {
                'donor_id': donor_id,
                'avg_donation': donor_history['donation_amount'].mean(),
                'recent_avg': rolling_avg.iloc[-6:].mean(),
                'growth_rate': growth_rate.iloc[-6:].mean(),
                'donation_std': donor_history['donation_amount'].std(),
                'months_active': (pd.to_datetime(donor_history['donation_date'].max()) - 
                                pd.to_datetime(donor_history['donation_date'].min())).days / 30,
                'total_donations': len(donor_history),
                'has_increased': (growth_rate.iloc[-6:] > 0).any(),
                'max_single_increase': growth_rate.max(),
                'recent_consistency': rolling_avg.iloc[-6:].std() / rolling_avg.iloc[-6:].mean()
            }

            donor_stats.append(latest_stats)

        features = pd.DataFrame(donor_stats)

        # Fill NaN values
        features = features.fillna(0)

        return features

    def train(self, features, donor_data):
        """
        Train the prediction model using actual historical upgrade patterns
        """
        try:
            # Create labels based on actual donation increases
            labels = []
            for donor_id in features['donor_id']:
                donor_history = donor_data[donor_data['donor_id'] == donor_id].sort_values('donation_date')

                # Check if donor has increased donations in the past
                initial_amount = donor_history['donation_amount'].iloc[0]
                recent_amount = donor_history['donation_amount'].iloc[-1]

                # Label as 1 if there's been a significant increase (>10%)
                labels.append(1 if (recent_amount > initial_amount * 1.1) else 0)

            labels = np.array(labels)

            # Ensure we have both classes represented
            if len(np.unique(labels)) < 2:
                print("Warning: Not enough variation in donor behavior for meaningful predictions")
                return {'precision': 0, 'recall': 0, 'roc_auc': 0.5}

            # Prepare features for training
            training_features = features.drop(['donor_id', 'has_increased'], axis=1)
            X = self.scaler.fit_transform(training_features)

            # Split and train
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.2, random_state=42, stratify=labels
            )

            self.model.fit(X_train, y_train)

            # Calculate metrics
            y_pred = self.model.predict(X_test)
            metrics = {
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred)
            }

            return metrics

        except Exception as e:
            print(f"Error during model training: {str(e)}")
            return {'precision': 0, 'recall': 0, 'roc_auc': 0.5}

    def predict(self, features):
        """
        Make predictions for donors
        """
        try:
            # Prepare features for prediction
            pred_features = features.drop(['donor_id', 'has_increased'], axis=1)
            X = self.scaler.transform(pred_features)

            # Get probabilities
            probabilities = self.model.predict_proba(X)

            return pd.DataFrame({
                'donor_id': features['donor_id'],
                'upgrade_probability': probabilities[:, 1]
            })

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return pd.DataFrame({
                'donor_id': features['donor_id'],
                'upgrade_probability': 0.5
            })