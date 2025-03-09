import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from donor_analysis import analyze_donors, calculate_upgrade_potential
from utils import load_and_validate_data, format_currency

# Page configuration
st.set_page_config(
    page_title="Donor Analysis Tool",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("🎯 Donor Analysis & Upgrade Potential Tool")
st.markdown("""
This tool helps analyze monthly donors and identify potential candidates for donation upgrades.
Upload your donor data to get started!
""")

# File upload
uploaded_file = st.file_uploader(
    "Upload your donor data (CSV)",
    type="csv",
    help="CSV should contain columns: donor_id, donation_amount, donation_date, email"
)

if uploaded_file is not None:
    try:
        # Load and validate data
        df = load_and_validate_data(uploaded_file)
        
        # Analysis section
        st.header("📊 Donor Analysis Dashboard")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Donors",
                len(df['donor_id'].unique())
            )
        
        with col2:
            avg_donation = df['donation_amount'].mean()
            st.metric(
                "Average Monthly Donation",
                format_currency(avg_donation)
            )
        
        with col3:
            total_monthly = df['donation_amount'].sum()
            st.metric(
                "Total Monthly Donations",
                format_currency(total_monthly)
            )
        
        with col4:
            median_donation = df['donation_amount'].median()
            st.metric(
                "Median Donation",
                format_currency(median_donation)
            )

        # Donation distribution
        st.subheader("📈 Donation Distribution")
        fig = px.histogram(
            df,
            x="donation_amount",
            nbins=30,
            title="Distribution of Monthly Donations"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Analyze donors and get upgrade potential
        donor_analysis = analyze_donors(df)
        upgrade_potential = calculate_upgrade_potential(donor_analysis)

        # Display upgrade candidates
        st.header("🎯 Upgrade Potential Analysis")

        # Add scoring criteria explanation
        with st.expander("ℹ️ How are upgrade scores calculated?"):
            st.markdown("""
            The upgrade score (0-100) is based on three factors:

            1. **Donation Consistency (40%)**: How stable are their donation amounts
                - More consistent donations = higher score

            2. **Duration of Support (30%)**: How long they've been donating
                - Score increases up to 12 months of giving

            3. **Donation Frequency (30%)**: How regularly they donate
                - Score increases up to 12 donations

            **Recommended Ask Amounts:**
            - Scores 80+: 50% increase suggested
            - Scores below 80: 25% increase suggested
            """)

        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            min_score = st.slider(
                "Minimum Upgrade Score",
                min_value=0,
                max_value=100,
                value=70,
                help="Filter donors by their upgrade potential score"
            )

        with col2:
            min_donations = st.number_input(
                "Minimum Donation Amount ($)",
                min_value=0,
                value=50,
                help="Filter donors by their current donation amount"
            )

        # Filter and display upgrade candidates
        upgrade_candidates = upgrade_potential[
            (upgrade_potential['upgrade_score'] >= min_score) &
            (upgrade_potential['current_donation'] >= min_donations)
        ].sort_values('upgrade_score', ascending=False)

        st.subheader("🌟 Recommended Upgrade Candidates")
        if not upgrade_candidates.empty:
            st.dataframe(
                upgrade_candidates[[
                    'donor_id',
                    'current_donation',
                    'donation_consistency',
                    'months_active',
                    'upgrade_score',
                    'recommended_ask'
                ]].style.format({
                    'current_donation': '${:.2f}',
                    'donation_consistency': '{:.1%}',
                    'upgrade_score': '{:.0f}',
                    'recommended_ask': '${:.2f}'
                })
            )

            # Export functionality
            csv = upgrade_candidates.to_csv(index=False)
            st.download_button(
                label="📥 Export Upgrade Candidates",
                data=csv,
                file_name="upgrade_candidates.csv",
                mime="text/csv"
            )
        else:
            st.info("No donors match the current filtering criteria.")

    except Exception as e:
        st.error(f"Error analyzing data: {str(e)}")
else:
    st.info("Please upload a CSV file to begin the analysis.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Made with ❤️ for charitable organizations</p>
</div>
""", unsafe_allow_html=True)