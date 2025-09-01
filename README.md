Customer Segmentation & Sales Analysis Project
Overview
This comprehensive analysis project explores customer behavior and sales patterns using an online retail dataset. The project includes:

Data preprocessing and cleaning

Exploratory data analysis (EDA) and sales trend visualization

Customer segmentation using RFM analysis

Sales forecasting using SARIMA models

Product recommendation strategy using association rule mining

Project Structure
text
customer_analysis_endtoend.ipynb  # Main analysis notebook
online_retail.csv                 # Dataset used for analysis
README.md                         # Project documentation
Key Features
Data Preprocessing: Robust handling of missing values, datetime conversion, and outlier detection

Sales Trend Analysis: Daily, weekly, and monthly sales visualization

Customer Segmentation: RFM (Recency, Frequency, Monetary) analysis with KMeans clustering

Sales Forecasting: Product and client-specific forecasting using SARIMA models

Recommendation Engine: Association rule mining for product recommendations

Business Insights: Actionable recommendations based on data analysis


Key Findings
Sales Trends:

Clear seasonal patterns with peak sales during holiday months

Weekly sales patterns showing higher activity on weekdays

Customer Segmentation:

Identified 4 distinct customer segments:

High-Value Loyalists (12.7% of customers)

Average Spenders (45.2%)

At-Risk Customers (25.1%)

Big Spenders (17.0%)

Regional Analysis:

Top 3 performing regions: United Kingdom, Germany, France

These regions contribute 78.3% of total sales

Product Associations:

Strongest association: PHOTO, VINTAGE → PACK (Lift: 8.52)

Other significant product relationships identified

Business Recommendations
Customer Retention:

Implement loyalty programs for High-Value Loyalists

Create win-back campaigns for At-Risk Customers

Regional Focus:

Allocate more marketing budget to UK and Germany

Investigate growth opportunities in underperforming regions

Inventory Management:

Increase stock before peak sales months (November-December)

Bundle frequently co-purchased products

Personalized Marketing:

Segment-specific promotions and communication

Recommendation engines for cross-selling opportunities

How to Run
Upload the online_retail.csv file to your Colab environment

Run the notebook cells sequentially

Adjust parameters as needed (sample size, forecasting periods, etc.)

Future Improvements
Incorporate more advanced forecasting models (Prophet, LSTM)

Implement collaborative filtering for recommendations

Add customer lifetime value (CLV) prediction

Create interactive dashboards for business users

Note
The association rule mining required parameter adjustments to generate meaningful results:

Used category-level aggregation due to high product diversity

Implemented adaptive parameter tuning to find optimal thresholds

Focused on multi-item transactions for better rule generation

For any questions or feedback, please open an issue in this repository.
