# ============================================================
# AI-POWERED URBAN TRAFFIC & PARKING INTELLIGENCE
# MODEL TRAINING SCRIPT
# ============================================================

import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# SCIKIT-LEARN
# ============================================================

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    silhouette_score
)

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans


# ============================================================
# TENSORFLOW / KERAS
# ============================================================

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LSTM
)

from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# CREATE MODELS DIRECTORY
# ============================================================

os.makedirs("models", exist_ok=True)


print("=" * 70)
print("AI-POWERED URBAN TRAFFIC & PARKING INTELLIGENCE")
print("MODEL TRAINING")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading datasets...")


traffic = pd.read_csv(r"C:\Users\senth\Desktop\PROJECT\AI_Traffic_Parking_Prediction\data\traffic_data_20000_2025.csv")
print('Traffic shape',traffic.shape)
weather = pd.read_csv(r"C:\Users\senth\Desktop\PROJECT\AI_Traffic_Parking_Prediction\data\weather_data_20000_2025.csv")
print('Weather shape',weather.shape)
parking = pd.read_csv(r"C:\Users\senth\Desktop\PROJECT\AI_Traffic_Parking_Prediction\data\parking_data_20000_2025.csv")
print('Parking shape',parking.shape)
Event = pd.read_csv(r"C:\Users\senth\Desktop\PROJECT\AI_Traffic_Parking_Prediction\data\event_data_2025.csv")
print('Event shape',Event.shape)
location = pd.read_csv(r"C:\Users\senth\Desktop\PROJECT\AI_Traffic_Parking_Prediction\data\location_data_2025.csv")
print('Loaction shape',location.shape)
Holiday  = pd.read_csv(r"C:\Users\senth\Desktop\PROJECT\AI_Traffic_Parking_Prediction\data\holiday_data_2025.csv")
print('Holiday shape',Holiday.shape)



# 2. MERGE TRAFFIC + LOCATION


print("\nMerging traffic and location data...")


df = traffic.merge(
    location,
    on="location_id",
    how="left"
)



# 3. MERGE WEATHER


print("Merging weather data...")


weather["date"] = pd.to_datetime(
    weather["date"]
)

traffic["date"] = pd.to_datetime(
    traffic["date"]
)


df["date"] = pd.to_datetime(
    df["date"]
)


# Merge using date + time

df = df.merge(
    weather[
        [
            "date",
            "time",
            "temperature",
            "rainfall",
            "humidity",
            "visibility"
        ]
    ],
    on=["date", "time"],
    how="left"
)


print("Final dataset:", df.shape)


# ============================================================
# 4. BASIC CLEANING
# ============================================================

print("\nCleaning data...")


# Remove duplicate rows

df = df.drop_duplicates()


# Convert numeric columns

numeric_columns = [
    "vehicle_count",
    "average_speed",
    "car_count",
    "bike_count",
    "bus_count",
    "truck_count",
    "road_capacity",
    "temperature",
    "rainfall",
    "humidity",
    "visibility"
]


for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# Fill missing numeric values

for col in numeric_columns:

    if col in df.columns:

        df[col] = df[col].fillna(
            df[col].median()
        )


# ============================================================
# 5. TIME FEATURE ENGINEERING
# ============================================================

print("Creating time features...")


df["hour"] = pd.to_datetime(
    df["time"],
    format="%H:%M",
    errors="coerce"
).dt.hour


df["day_of_week"] = df["date"].dt.dayofweek


df["month"] = df["date"].dt.month


# Weekend

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


# Peak hours

df["is_peak_hour"] = (
    df["hour"].isin(
        [7, 8, 9, 17, 18, 19]
    )
).astype(int)


# ============================================================
# 6. ENCODE CONGESTION LEVEL
# ============================================================

print("Encoding congestion level...")


label_encoder = LabelEncoder()


df["congestion_encoded"] = (
    label_encoder.fit_transform(
        df["congestion_level"]
    )
)


print(
    "\nCongestion classes:"
)

for i, label in enumerate(
    label_encoder.classes_
):

    print(
        i,
        "->",
        label
    )


# Save encoder

with open(
    "models/congestion_encoder.pkl",
    "wb"
) as f:

    pickle.dump(
        label_encoder,
        f
    )


# ============================================================
# 7. FEATURE SELECTION
# ============================================================

features = [
    "average_speed",
    "car_count",
    "bike_count",
    "bus_count",
    "truck_count",
    "road_capacity",
    "temperature",
    "rainfall",
    "humidity",
    "visibility",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour"
]


# Keep only available columns

features = [
    col
    for col in features
    if col in df.columns
]


print("\nFeatures used:")

for feature in features:

    print("-", feature)


X = df[features].copy()


# ============================================================
# TARGET 1
# VEHICLE COUNT
# ============================================================

y_traffic = df[
    "vehicle_count"
]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_traffic,

    test_size=0.20,

    random_state=42
)


# ============================================================
# STANDARD SCALER
# ============================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)


# Save scaler

with open(
    "models/traffic_scaler.pkl",
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )


# ============================================================
# MODEL RESULTS
# ============================================================

results = {}


# ============================================================
# 8. LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 70)

print("1. LINEAR REGRESSION")

print("=" * 70)


linear_model = LinearRegression()


linear_model.fit(
    X_train_scaled,
    y_train
)


linear_pred = linear_model.predict(
    X_test_scaled
)


linear_mae = mean_absolute_error(
    y_test,
    linear_pred
)


linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_pred
    )
)


linear_r2 = r2_score(
    y_test,
    linear_pred
)


print(
    "MAE :",
    round(linear_mae, 2)
)

print(
    "RMSE:",
    round(linear_rmse, 2)
)

print(
    "R2  :",
    round(linear_r2, 4)
)


results["linear_regression"] = {

    "model": linear_model,

    "type": "regression",

    "name": "Linear Regression",

    "mae": round(
        linear_mae,
        2
    ),

    "rmse": round(
        linear_rmse,
        2
    ),

    "r2": round(
        linear_r2,
        4
    )
}


# ============================================================
# 9. CONGESTION DATA
# ============================================================

y_congestion = df[
    "congestion_encoded"
]


Xc_train, Xc_test, yc_train, yc_test = train_test_split(

    X,

    y_congestion,

    test_size=0.20,

    random_state=42,

    stratify=y_congestion
)


Xc_train_scaled = scaler.fit_transform(
    Xc_train
)


Xc_test_scaled = scaler.transform(
    Xc_test
)


# ============================================================
# 10. DECISION TREE
# ============================================================

print("\n" + "=" * 70)

print("2. DECISION TREE")

print("=" * 70)


decision_tree = DecisionTreeClassifier(

    max_depth=8,

    random_state=42
)


decision_tree.fit(
    Xc_train,
    yc_train
)


dt_pred = decision_tree.predict(
    Xc_test
)


dt_accuracy = accuracy_score(
    yc_test,
    dt_pred
)


print(
    "Accuracy:",
    round(
        dt_accuracy * 100,
        2
    ),
    "%"
)


print("\nClassification Report:")

print(
    classification_report(
        yc_test,
        dt_pred,
        target_names=label_encoder.classes_
    )
)


results["decision_tree"] = {

    "model": decision_tree,

    "type": "classification",

    "name": "Decision Tree",

    "accuracy": round(
        dt_accuracy * 100,
        2
    )
}


# ============================================================
# 11. RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)

print("3. RANDOM FOREST")

print("=" * 70)


random_forest = RandomForestClassifier(

    n_estimators=200,

    max_depth=12,

    random_state=42,

    n_jobs=-1
)


random_forest.fit(
    Xc_train,
    yc_train
)


rf_pred = random_forest.predict(
    Xc_test
)


rf_accuracy = accuracy_score(
    yc_test,
    rf_pred
)


print(
    "Accuracy:",
    round(
        rf_accuracy * 100,
        2
    ),
    "%"
)


print("\nClassification Report:")

print(
    classification_report(
        yc_test,
        rf_pred,
        target_names=label_encoder.classes_
    )
)


results["random_forest"] = {

    "model": random_forest,

    "type": "classification",

    "name": "Random Forest",

    "accuracy": round(
        rf_accuracy * 100,
        2
    )
}


# ============================================================
# 12. K-MEANS CLUSTERING
# ============================================================

print("\n" + "=" * 70)

print("4. K-MEANS")

print("=" * 70)


cluster_features = [

    "vehicle_count",

    "average_speed",

    "car_count",

    "bike_count",

    "bus_count",

    "truck_count"

]


cluster_features = [

    col

    for col in cluster_features

    if col in df.columns

]


cluster_data = df[
    cluster_features
].copy()


cluster_scaler = StandardScaler()


cluster_scaled = cluster_scaler.fit_transform(
    cluster_data
)


kmeans = KMeans(

    n_clusters=3,

    random_state=42,

    n_init=10
)


clusters = kmeans.fit_predict(
    cluster_scaled
)


silhouette = silhouette_score(
    cluster_scaled,
    clusters
)


print(
    "Clusters:",
    3
)

print(
    "Silhouette Score:",
    round(
        silhouette,
        4
    )
)


results["kmeans"] = {

    "model": kmeans,

    "type": "clustering",

    "name": "K-Means",

    "silhouette_score": round(
        silhouette,
        4
    )
}


# Save cluster scaler

with open(
    "models/kmeans_scaler.pkl",
    "wb"
) as f:

    pickle.dump(
        cluster_scaler,
        f
    )


# ============================================================
# 13. ANN
# ============================================================

print("\n" + "=" * 70)

print("5. ARTIFICIAL NEURAL NETWORK")

print("=" * 70)


ann_model = Sequential([

    Dense(
        128,
        activation="relu",
        input_shape=(
            X_train_scaled.shape[1],
        )
    ),

    Dropout(0.2),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1,
        activation="linear"
    )

])


ann_model.compile(

    optimizer="adam",

    loss="mse",

    metrics=["mae"]
)


early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True
)


ann_history = ann_model.fit(

    X_train_scaled,

    y_train,

    validation_split=0.2,

    epochs=100,

    batch_size=32,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


ann_pred = ann_model.predict(
    X_test_scaled,
    verbose=0
).flatten()


ann_mae = mean_absolute_error(
    y_test,
    ann_pred
)


ann_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        ann_pred
    )
)


ann_r2 = r2_score(
    y_test,
    ann_pred
)


print(
    "\nANN MAE :",
    round(ann_mae, 2)
)

print(
    "ANN RMSE:",
    round(ann_rmse, 2)
)

print(
    "ANN R2  :",
    round(ann_r2, 4)
)


results["ann"] = {

    "model": ann_model,

    "type": "deep_learning",

    "name": "Artificial Neural Network",

    "mae": round(
        ann_mae,
        2
    ),

    "rmse": round(
        ann_rmse,
        2
    ),

    "r2": round(
        ann_r2,
        4
    )
}


# ============================================================
# 14. LSTM DATA PREPARATION
# ============================================================

print("\n" + "=" * 70)

print("6. LSTM")

print("=" * 70)


# LSTM will use historical vehicle counts
# to predict the next vehicle count.


lstm_data = df[
    [
        "date",
        "time",
        "vehicle_count"
    ]
].copy()


lstm_data["datetime"] = pd.to_datetime(

    lstm_data["date"].astype(str)
    + " "
    + lstm_data["time"].astype(str),

    errors="coerce"
)


lstm_data = lstm_data.sort_values(
    "datetime"
)


lstm_values = lstm_data[
    "vehicle_count"
].values.reshape(-1, 1)


lstm_scaler = StandardScaler()


lstm_scaled = lstm_scaler.fit_transform(
    lstm_values
)


# ============================================================
# CREATE SEQUENCES
# ============================================================

sequence_length = 12


X_lstm = []

y_lstm = []


for i in range(
    sequence_length,
    len(lstm_scaled)
):

    X_lstm.append(

        lstm_scaled[
            i - sequence_length:i
        ]

    )

    y_lstm.append(
        lstm_scaled[i]
    )


X_lstm = np.array(
    X_lstm
)


y_lstm = np.array(
    y_lstm
)


print(
    "LSTM X shape:",
    X_lstm.shape
)

print(
    "LSTM y shape:",
    y_lstm.shape
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

split_index = int(
    len(X_lstm) * 0.80
)


X_lstm_train = X_lstm[
    :split_index
]


X_lstm_test = X_lstm[
    split_index:
]


y_lstm_train = y_lstm[
    :split_index
]


y_lstm_test = y_lstm[
    split_index:
]


# ============================================================
# BUILD LSTM
# ============================================================

lstm_model = Sequential([

    LSTM(
        64,
        return_sequences=True,
        input_shape=(
            X_lstm_train.shape[1],
            X_lstm_train.shape[2]
        )
    ),

    Dropout(0.2),

    LSTM(
        32
    ),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1
    )

])


lstm_model.compile(

    optimizer="adam",

    loss="mse",

    metrics=["mae"]

)


lstm_history = lstm_model.fit(

    X_lstm_train,

    y_lstm_train,

    validation_split=0.2,

    epochs=50,

    batch_size=32,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# LSTM PREDICTION
# ============================================================

lstm_pred_scaled = lstm_model.predict(

    X_lstm_test,

    verbose=0

)


lstm_pred = lstm_scaler.inverse_transform(

    lstm_pred_scaled

).flatten()


lstm_actual = lstm_scaler.inverse_transform(

    y_lstm_test

).flatten()


lstm_mae = mean_absolute_error(

    lstm_actual,

    lstm_pred

)


lstm_rmse = np.sqrt(

    mean_squared_error(

        lstm_actual,

        lstm_pred

    )

)


lstm_r2 = r2_score(

    lstm_actual,

    lstm_pred

)


print(
    "\nLSTM MAE :",
    round(lstm_mae, 2)
)

print(
    "LSTM RMSE:",
    round(lstm_rmse, 2)
)

print(
    "LSTM R2  :",
    round(lstm_r2, 4)
)


results["lstm"] = {

    "model": lstm_model,

    "type": "deep_learning",

    "name": "LSTM",

    "mae": round(
        lstm_mae,
        2
    ),

    "rmse": round(
        lstm_rmse,
        2
    ),

    "r2": round(
        lstm_r2,
        4
    )

}


# ============================================================
# 15. SAVE LSTM SCALER
# ============================================================

with open(

    "models/lstm_scaler.pkl",

    "wb"

) as f:

    pickle.dump(

        lstm_scaler,

        f

    )


# ============================================================
# 16. SAVE ML MODELS
# ============================================================

print("\nSaving ML models...")


with open(

    "models/linear_regression.pkl",

    "wb"

) as f:

    pickle.dump(

        linear_model,

        f

    )


with open(

    "models/decision_tree.pkl",

    "wb"

) as f:

    pickle.dump(

        decision_tree,

        f

    )


with open(

    "models/random_forest.pkl",

    "wb"

) as f:

    pickle.dump(

        random_forest,

        f

    )


with open(

    "models/kmeans.pkl",

    "wb"

) as f:

    pickle.dump(

        kmeans,

        f

    )


# ============================================================
# 17. SAVE DEEP LEARNING MODELS
# ============================================================

print("Saving deep learning models...")


ann_model.save(
    "models/ann_model.keras"
)


lstm_model.save(
    "models/lstm_model.keras"
)


# ============================================================
# 18. SAVE FEATURE INFORMATION
# ============================================================

with open(

    "models/features.pkl",

    "wb"

) as f:

    pickle.dump(

        features,

        f

    )


# ============================================================
# 19. SAVE COMPLETE MODEL INFORMATION
# ============================================================

model_information = {

    "linear_regression": {

        "name":
        "Linear Regression",

        "purpose":
        "Predict vehicle count",

        "type":
        "Regression",

        "mae":
        linear_mae,

        "rmse":
        linear_rmse,

        "r2":
        linear_r2

    },

    "decision_tree": {

        "name":
        "Decision Tree",

        "purpose":
        "Predict congestion level",

        "type":
        "Classification",

        "accuracy":
        dt_accuracy

    },

    "random_forest": {

        "name":
        "Random Forest",

        "purpose":
        "Predict congestion level",

        "type":
        "Classification",

        "accuracy":
        rf_accuracy

    },

    "kmeans": {

        "name":
        "K-Means",

        "purpose":
        "Identify traffic patterns",

        "type":
        "Clustering",

        "silhouette_score":
        silhouette

    },

    "ann": {

        "name":
        "Artificial Neural Network",

        "purpose":
        "Predict vehicle count",

        "type":
        "Deep Learning",

        "mae":
        ann_mae,

        "rmse":
        ann_rmse,

        "r2":
        ann_r2

    },

    "lstm": {

        "name":
        "LSTM",

        "purpose":
        "Predict future traffic",

        "type":
        "Deep Learning",

        "mae":
        lstm_mae,

        "rmse":
        lstm_rmse,

        "r2":
        lstm_r2

    }

}


with open(

    "models/model_information.pkl",

    "wb"

) as f:

    pickle.dump(

        model_information,

        f

    )


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n")

print("=" * 70)

print("TRAINING COMPLETED SUCCESSFULLY")

print("=" * 70)


print("\nSaved files:")


for file in os.listdir(
    "models"
):

    print(
        "✓",
        file
    )


print("\n" + "=" * 70)

print("MODEL SUMMARY")

print("=" * 70)


print(
    f"""
Linear Regression
    MAE  : {linear_mae:.2f}
    RMSE : {linear_rmse:.2f}
    R2   : {linear_r2:.4f}

Decision Tree
    Accuracy : {dt_accuracy * 100:.2f}%

Random Forest
    Accuracy : {rf_accuracy * 100:.2f}%

K-Means
    Silhouette Score : {silhouette:.4f}

ANN
    MAE  : {ann_mae:.2f}
    RMSE : {ann_rmse:.2f}
    R2   : {ann_r2:.4f}

LSTM
    MAE  : {lstm_mae:.2f}
    RMSE : {lstm_rmse:.2f}
    R2   : {lstm_r2:.4f}
"""
)

