import pandas as pd
import numpy as np
from datetime import datetime

def analyze_donors(df):
    """
    Analyze donor data and calculate key metrics
    """
    # Group by donor
    donor_analysis = df.groupby('donor_id').agg({
        'donation_amount': ['mean', 'std', 'count'],
        'donation_date': ['min', 'max']
    }).reset_index()

    # Flatten column names
    donor_analysis.columns = [
        'donor_id',
        'avg_donation',
        'std_donation',
        'donation_count',
        'first_donation',
        'last_donation'
    ]

    # Calculate additional metrics
    donor_analysis['months_active'] = (
        (pd.to_datetime(donor_analysis['last_donation']) -
         pd.to_datetime(donor_analysis['first_donation'])).dt.days / 30
    ).round()

    # Calculate donation consistency (std dev relative to mean)
    donor_analysis['donation_consistency'] = 1 - (
        donor_analysis['std_donation'] / donor_analysis['avg_donation']
    ).fillna(0)
    
    # Clip consistency to 0-1 range
    donor_analysis['donation_consistency'] = donor_analysis['donation_consistency'].clip(0, 1)

    return donor_analysis

def calculate_upgrade_potential(donor_analysis):
    """
    Calculate upgrade potential scores and recommendations
    """
    # Initialize upgrade potential dataframe
    upgrade_potential = donor_analysis.copy()

    # Calculate upgrade score (0-100 scale)
    upgrade_potential['upgrade_score'] = (
        # Consistency weight (40%)
        (donor_analysis['donation_consistency'] * 40) +
        # Months active weight (30%)
        (np.minimum(donor_analysis['months_active'] / 12, 1) * 30) +
        # Donation frequency weight (30%)
        (np.minimum(donor_analysis['donation_count'] / 12, 1) * 30)
    )

    # Calculate recommended ask amount
    upgrade_potential['current_donation'] = donor_analysis['avg_donation']
    upgrade_potential['recommended_ask'] = np.where(
        upgrade_potential['upgrade_score'] >= 80,
        donor_analysis['avg_donation'] * 1.5,  # 50% increase for high-score donors
        donor_analysis['avg_donation'] * 1.25   # 25% increase for others
    )

    # Round scores and amounts
    upgrade_potential['upgrade_score'] = upgrade_potential['upgrade_score'].round(1)
    upgrade_potential['recommended_ask'] = upgrade_potential['recommended_ask'].round(2)

    return upgrade_potential
