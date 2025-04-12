import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pickle
import warnings
from pandas.errors import PerformanceWarning
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import os


# Suppress PerformanceWarning specifically
warnings.filterwarnings("ignore", category=PerformanceWarning)



# Load training dataset
df = pd.read_json("fullDataSet.json", lines=True)

# Compose timestamp from one-hot encoded week, day, hour
week_cols = [col for col in df.columns if col.startswith("week_")]
day_cols = [col for col in df.columns if col.startswith("day_")]
hour_cols = [col for col in df.columns if col.startswith("ishour")]

df["week"] = df[week_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)
df["day"] = df[day_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)
df["hour"] = df[hour_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)

# Base date
def base_timestamp(df):
    base_date = pd.Timestamp("2024-01-01")
    dt = (
        base_date +
        pd.to_timedelta(df["week"] * 7, unit="D") +
        pd.to_timedelta(df["day"], unit="D") +
        pd.to_timedelta(df["hour"], unit="h")
    )
    return (dt - base_date).dt.total_seconds() / 3600

# Perform correlation analysis (We start the timestamp from start of 2024 since that is where the bookings started)
df["timestamp"] = base_timestamp(df)
df["delivery_datetime"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(df["timestamp"], unit="h")

# Drop negatively correlated features (including time one-hots)
time_cols = week_cols + day_cols + hour_cols + ["week", "day", "hour"]
correlation_matrix = df.drop(columns=["timestamp", "delivery_datetime"]).copy()
correlation_matrix["timestamp"] = df["timestamp"]
corr_result = correlation_matrix.corr()["timestamp"].drop("timestamp")
negative_corr_features = corr_result[corr_result < 0].index.tolist()

df = df.drop(columns=negative_corr_features)

with open("negatively_correlated_features.pkl", "wb") as f:
    pickle.dump(negative_corr_features, f)


# Define features and target
target_col = "timestamp"
X = df.drop(columns=[target_col, "delivery_datetime", "week", "day", "hour"], errors="ignore")
y = df[target_col]

# K-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = []
rmse_scores = []

print("Starting K-Fold training...")
for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    print(f"\nFold {fold}")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = Ridge()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R²: {r2:.4f}, RMSE: {rmse:.2f} hours")
    r2_scores.append(r2)
    rmse_scores.append(rmse)

print("\nFinal K-Fold Results:")
print(f"Avg R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Avg RMSE: {np.mean(rmse_scores):.2f} ± {np.std(rmse_scores):.2f} hours")

print("\nTraining final model on full dataset and saving it...")
final_model = Ridge()
final_model.fit(X, y)
model_path = "final_delivery_model.pkl"
joblib.dump(final_model, model_path)
print(f"Model saved to {model_path}")





# Load the final model and negatively correlated features
final_model = joblib.load("final_delivery_model.pkl")
negative_corr_features = joblib.load("negatively_correlated_features.pkl")

# Load the bookings dataframe
bookings_df = pd.read_json("output_validation.json", lines=True)

# Load customer columns and location columns from pickle
with open("customer_columns.pkl", "rb") as f:
    customer_columns = pickle.load(f)

with open("location_columns.pkl", "rb") as f:
    location_columns = pickle.load(f)

# We need to add the customer columns from the training data and fill it with 0's
# Ensure customer columns are added, defaulting to 0 if not already present
for customer_col in customer_columns:
    if customer_col not in bookings_df.columns:
        bookings_df[customer_col] = 0  # Add column with default value 0
    else:
        # If column exists and has a 1, keep it as 1, otherwise leave as it is
        bookings_df[customer_col] = bookings_df[customer_col].apply(lambda x: 1 if x == 1 else x)

# Ensure location columns are added, defaulting to 0 if not already present
for location_col in location_columns:
    if location_col not in bookings_df.columns:
        bookings_df[location_col] = 0  # Add column with default value 0
    else:
        # If column exists and has a 1, keep it as 1, otherwise leave as it is
        bookings_df[location_col] = bookings_df[location_col].apply(lambda x: 1 if x == 1 else x)

trained_features = final_model.feature_names_in_

# Drop columns from bookings_df that are not in the trained features
columns_to_drop = [col for col in bookings_df.columns if col not in trained_features]

# Drop those columns from the bookings_df
bookings_df = bookings_df.drop(columns=columns_to_drop, errors='ignore')




# Remove negatively correlated features
bookings_df = bookings_df.drop(columns=[col for col in negative_corr_features if col in bookings_df.columns], errors="ignore")
bookings_df = bookings_df[trained_features]

bookings_df.to_json("bookings_df.json", orient="records", lines=True)

# Predict timestamp and convert to datetime
predicted_ts = final_model.predict(bookings_df)
predicted_dt = pd.Timestamp("2024-01-01") + pd.to_timedelta(predicted_ts, unit="h")

bookings_df["predicted_delivery_datetime"] = predicted_dt



# Save the predictions to a new JSON file
bookings_df.to_json("predicted_bookings.json", orient="records", lines=True)


pred_df = pd.read_json("predicted_bookings.json", lines=True)
meta_df = pd.read_json("validationSetWihoutDatetime.json", lines=True)
first_leg_df = pd.read_json("found_routes_with_times_first_two.json")
second_leg_df = pd.read_json("found_routes_with_times_second_two.json")

# Store adjusted predictions
adjusted_predictions = []


# Adjusting the prediction results to fit within a time window from the booking.
for i, row in pred_df.iterrows():
    meta_row = meta_df.iloc[i]
    original_pred = pd.to_datetime(row["predicted_delivery_datetime"])

    # Determine window_start
    if pd.notnull(meta_row.get("first_pickup")):
        window_start = pd.to_datetime(meta_row["first_pickup"])
    elif pd.notnull(meta_row.get("last_pickup")):
        window_start = pd.to_datetime(meta_row["last_pickup"])
    elif pd.notnull(meta_row.get("cargo_closing")):
        window_end = pd.to_datetime(meta_row["cargo_closing"])
        window_start = window_end - timedelta(days=9)
    elif pd.notnull(meta_row.get("cargo_opening")):
        window_end = pd.to_datetime(meta_row["cargo_opening"])
        window_start = window_end - timedelta(days=9)
    else:
        adjusted_predictions.append(original_pred)
        continue

    # Determine window_end
    if pd.notnull(meta_row.get("cargo_closing")):
        window_end = pd.to_datetime(meta_row["cargo_closing"])
    elif pd.notnull(meta_row.get("cargo_opening")):
        window_end = pd.to_datetime(meta_row["cargo_opening"])
    else:
        window_end = window_start + timedelta(days=9)

    # Calculate midpoint and shift
    midpoint = (window_start + (window_end - window_start) / 2) + timedelta(days=3)
    hour = original_pred.hour + 8

    adjusted_pred = pd.Timestamp(
        year=midpoint.year,
        month=midpoint.month,
        day=midpoint.day,
        hour=hour,
        minute=original_pred.minute,
        second=original_pred.second
    )

    adjusted_predictions.append(adjusted_pred)

# Add adjusted prediction
pred_df["adjusted_predicted_delivery_datetime"] = adjusted_predictions

# Use durations from both legs
first_leg_durations = first_leg_df["average_time_minutes"].values
second_leg_durations = second_leg_df["average_time_minutes"].values

# Calculate estimated departure and arrival
estimated_departures = [
    adjusted - timedelta(minutes=first_leg)
    for adjusted, first_leg in zip(adjusted_predictions, first_leg_durations)
]
estimated_arrivals = [
    adjusted + timedelta(minutes=second_leg)
    for adjusted, second_leg in zip(adjusted_predictions, second_leg_durations)
]

# Assign new columns
pred_df.insert(0, "booking_id", second_leg_df["booking_id"].values)  # Insert id at beginning
pred_df["estimated_departure"] = estimated_departures
pred_df["estimated_arrival"] = estimated_arrivals

pred_df.drop(columns=["predicted_delivery_datetime"], inplace=True)


# Save results
pred_df.to_json("adjusted_predicted_bookings.json", orient="records", lines=True)

print("adjusted_predicted_bookings.json created")

# Load the JSON file
df = pd.read_json('adjusted_predicted_bookings.json', lines=True)

# Calculate mean, median and graph out results

validation_df = pd.read_json("validationSetWithDatetime.json")
predicted_df = pd.read_json("predicted_bookings.json", lines=True)
predicted_df = pd.read_json("adjusted_predicted_bookings.json", lines=True)

# Convert 'delivery_datetime' in validation_df to Unix timestamp (in seconds)
validation_df["delivery_datetime_unix"] = validation_df["delivery_datetime"].apply(
    lambda x: pd.to_datetime(x).timestamp() if pd.notnull(x) else None
)

# Compare rows 1 and 1, 2 and 2, etc.
comparison_results = []
for idx in range(min(len(validation_df), len(predicted_df))):
    # Get the corresponding row from both dataframes
    validation_row = validation_df.iloc[idx]
    predicted_row = predicted_df.iloc[idx]

    # Compare the Unix timestamps 
    validation_unix = validation_row["delivery_datetime_unix"]

    predicted_unix = predicted_row["adjusted_predicted_delivery_datetime"] / 1000  # Convert milliseconds to seconds

    # Compute the difference in days
    if pd.notnull(validation_unix) and pd.notnull(predicted_unix):
        diff_in_days = (predicted_unix - validation_unix) / (60 * 60 * 24)  # convert seconds to days

        # Calculate time of day difference
        validation_time = pd.to_datetime(validation_unix, unit="s").time()
        predicted_time = pd.to_datetime(predicted_unix, unit="s").time()

        # Calculate time difference in seconds
        time_diff_in_seconds = (pd.to_datetime(predicted_time.strftime('%H:%M:%S'), format='%H:%M:%S') - 
                                pd.to_datetime(validation_time.strftime('%H:%M:%S'), format='%H:%M:%S')).total_seconds()

        # Convert time difference to hours, minutes, seconds
        time_diff_in_hours = time_diff_in_seconds // 3600
        time_diff_in_minutes = (time_diff_in_seconds % 3600) // 60
        time_diff_in_seconds_remaining = time_diff_in_seconds % 60

    else:
        diff_in_days = None
        time_diff_in_hours = None
        time_diff_in_minutes = None
        time_diff_in_seconds_remaining = None

    # Store comparison results
    comparison_results.append({
        "booking_id": validation_row["booking_id"],
        "predicted_delivery_datetime": predicted_unix,
        "actual_delivery_datetime": validation_unix,
        "difference_in_days": diff_in_days,
        "time_diff_in_hours": time_diff_in_hours,
        "time_diff_in_minutes": time_diff_in_minutes,
        "time_diff_in_seconds": time_diff_in_seconds_remaining
    })

# Convert results to a df and save
comparison_df = pd.DataFrame(comparison_results)

comparison_df.to_json("comparison_results.json", orient="records", lines=True)

# Total number of records before filtering
total_records = len(comparison_df)

# Filter for the difference in days within the range [-7, 7]
filtered_comparison_df = comparison_df[
    comparison_df["difference_in_days"].between(-7, 7, inclusive='both')
]

# Number of records after filtering
filtered_records = len(filtered_comparison_df)

# Calculate percentage of records that were excluded (outside the -7 to 7 day range)
excluded_percentage = 100 * (total_records - filtered_records) / total_records

# Calculate median, mean, and std deviation differences for the filtered data
median_days = filtered_comparison_df["difference_in_days"].median()
mean_days = filtered_comparison_df["difference_in_days"].mean()
std_days = filtered_comparison_df["difference_in_days"].std()

median_hours = filtered_comparison_df["time_diff_in_hours"].median()
mean_hours = filtered_comparison_df["time_diff_in_hours"].mean()
std_hours = filtered_comparison_df["time_diff_in_hours"].std()

mse_days = ((comparison_df["difference_in_days"].dropna())**2).mean()
mse_hours = ((comparison_df["time_diff_in_hours"].dropna())**2).mean()

# Print statistics
print(f"Median difference in days: {median_days:.2f}")
print(f"Mean difference in days: {mean_days:.2f}")
print(f"Standard Deviation in days: {std_days:.2f}")

print(f"Median time difference in hours: {median_hours:.2f}")
print(f"Mean time difference in hours: {mean_hours:.2f}")
print(f"Standard Deviation in hours: {std_hours:.2f}")


print(f"Mean Squared Error (MSE) for Days: {mse_days:.2f}")
print(f"Mean Squared Error (MSE) for Hours: {mse_hours:.2f}")
# Print the percentage of records excluded from the graph
print(f"Percentage of records excluded (outside -7 to 7 days): {excluded_percentage:.2f}%")


sns.set(style="whitegrid")

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

# Plot 1: Difference in Days (filtered)
sns.histplot(filtered_comparison_df["difference_in_days"].dropna(), bins=15, kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Prediction Error: Days")
axes[0].set_xlabel("Difference in Days")
axes[0].set_ylabel("Count")
axes[0].set_xticks(range(-7, 8))  # Set integer ticks from -7 to 7

# Plot 2: Time Difference in Hours (filtered)
sns.histplot(filtered_comparison_df["time_diff_in_hours"].dropna(), bins=30, kde=True, ax=axes[1], color="salmon")
axes[1].set_title("Prediction Error: Time of Day (Hours)")
axes[1].set_xlabel("Difference in Hours")
axes[1].set_ylabel("Count")
axes[1].set_xticks(range(-7, 8))  # Set integer ticks from -7 to 7

# Adjust layout and show
plt.tight_layout()
plt.show()

# Print comparison for the first few rows
print(filtered_comparison_df.head())


# Create final output with the delivery_datetime, and estimated arrival, departure, booking_id

df = pd.read_json("adjusted_predicted_bookings.json", lines=True)
df = df[["adjusted_predicted_delivery_datetime", "estimated_departure", "estimated_arrival", "booking_id"]]

df["adjusted_predicted_delivery_datetime"] = pd.to_datetime(df["adjusted_predicted_delivery_datetime"], unit="ms", utc=True)
df["adjusted_predicted_delivery_datetime"] = df["adjusted_predicted_delivery_datetime"].dt.tz_convert('Europe/Amsterdam').dt.strftime('%Y-%m-%d %H:%M')

df["estimated_departure"] = pd.to_datetime(df["estimated_departure"], unit="ms", utc=True)
df["estimated_departure"] = df["estimated_departure"].dt.tz_convert('Europe/Amsterdam').dt.strftime('%Y-%m-%d %H:%M')

df["estimated_arrival"] = pd.to_datetime(df["estimated_arrival"], unit="ms", utc=True)
df["estimated_arrival"] = df["estimated_arrival"].dt.tz_convert('Europe/Amsterdam').dt.strftime('%Y-%m-%d %H:%M')

df.to_json("converted_adjusted_predicted_bookings.json", orient="records", lines=True)

print("Timestamps converted and saved to 'converted_adjusted_predicted_bookings.json'")

# Cleanup unused files

file_paths = []

file_paths = 'bookings_df.json', 'fixed_encoding.json', "found_routes_with_times_first_two.json", 'found_routes_with_times_second_two.json', 'fullDataSet.json', 'output_validation.json', 'output_validation.json', 'output_with_date_column.json', 'output_with_week_day_hour_customer_location.json', 'output.json', 'predicted_bookings.json', 'processed_bookings_full.json','route_averages_first_two.json', 'route_averages_second_two.json', 'temp_times_second_two.json', 'temp_times.json', 'validationSetWihoutDatetime.json', 'validationSetWithDatetime.json' 

for file_path in file_paths:
    # Delete the file
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} has been deleted.")
    else:
        print(f"File {file_path} does not exist.")