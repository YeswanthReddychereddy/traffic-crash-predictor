
import pandas as pd
import plotly.express as px
from preswald import connect, get_df, table, text, plotly, slider
from sklearn.ensemble import RandomForestClassifier

# Initialize connection to Preswald service
connect()

# Path to your dataset (this should be set in your preswald.toml as well)
data_file = "data/traffic_crashes/Traffic_Crashes_Crashes.csv"

# Load data using Preswald (if needed)
df = get_df("traffic_crashes")

# Ensure the dataframe is loaded correctly
if df is not None:
    text("Data loaded successfully!")

    # Data Cleaning and Preprocessing (handle NaN values and prepare for modeling)
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'POSTED_SPEED_LIMIT', 'LANE_CNT', 'MOST_SEVERE_INJURY'])

    # EDA (Exploratory Data Analysis)
    # Scatter plot of traffic crashes by location and speed limit
    fig = px.scatter(df, x="LATITUDE", y="LONGITUDE", color="POSTED_SPEED_LIMIT", title="Traffic Crashes by Location and Speed Limit")
    plotly(fig)

    # Bar plot showing crash severity distribution
    severity_counts = df['MOST_SEVERE_INJURY'].value_counts().reset_index()
    severity_counts.columns = ['Severity', 'Count']
    fig2 = px.bar(severity_counts, x='Severity', y='Count', title="Crash Severity Distribution")
    plotly(fig2)

    # Prepare features for machine learning (Use relevant columns only)
    X = df[['LATITUDE', 'LONGITUDE', 'POSTED_SPEED_LIMIT', 'LANE_CNT']]
    y = df['MOST_SEVERE_INJURY']

    # Train the RandomForest model (fit once for the session)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Get user inputs through sliders
    latitude_input = slider("Latitude", min_val=-90, max_val=90, default=41.816073)
    longitude_input = slider("Longitude", min_val=-180, max_val=180, default=-87.656743)
    speed_limit_input = slider("Speed Limit", min_val=0, max_val=100, default=30)
    lane_count_input = slider("Lane Count", min_val=1, max_val=6, default=3)

    # Prepare the input for prediction in the same format as training
    user_input = pd.DataFrame([[latitude_input, longitude_input, speed_limit_input, lane_count_input]],
                              columns=['LATITUDE', 'LONGITUDE', 'POSTED_SPEED_LIMIT', 'LANE_CNT'])

    # Make the prediction
    try:
        predicted_severity = model.predict(user_input)
        text(f"Predicted severity of crash based on inputs: {predicted_severity[0]}")
    except Exception as e:
        text(f"Error during prediction: {str(e)}")

else:
    text("Failed to load data!")

