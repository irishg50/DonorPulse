import pandas as pd
from datetime import datetime

def load_and_validate_data(file):
    """
    Load and validate CSV data
    """
    try:
        df = pd.read_csv(file)
        
        # Check required columns
        required_columns = ['donor_id', 'donation_amount', 'donation_date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {', '.join(missing_columns)}"
            )

        # Convert dates
        df['donation_date'] = pd.to_datetime(df['donation_date'])
        
        # Validate donation amounts
        if (df['donation_amount'] < 0).any():
            raise ValueError("Negative donation amounts found")

        # Sort by date
        df = df.sort_values('donation_date')
        
        return df

    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty")
    except pd.errors.ParserError:
        raise ValueError("Unable to parse the CSV file. Please check the format")

def format_currency(amount):
    """
    Format amount as currency
    """
    return f"${amount:,.2f}"
