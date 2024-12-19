import os
# import ast
# import random
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import pydeck as pdk
import altair as alt
import matplotlib.pyplot as plt

from core import *


import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import requests


switch_page = st.button("Switch page")
if switch_page:
    st.switch_page('ui_visualize.py')


# Sidebar: File upload
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Placeholder for data
data = None
model = None

features_list = ["primary_label", "latitude", "longitude", "scientific_name", "common_name", "date", "season"]


def validate_file(file):
    try:
        df = pd.read_csv(file)
        if not all(col in df.columns for col in features_list):
            st.error("The file must contain the following columns: " + ", ".join(features_list))
            return None
        return df
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None


def recognize_classes(df, model):
    missing_indices = df[df['primary_label'].isna()].index
    total = len(missing_indices)

    if total == 0:
        st.success("No missing values in primary_label column.")
        return df

    progress_bar = st.progress(0)
    recognized_classes = []

    for i, idx in enumerate(missing_indices):
        # Simulate model prediction
        row_features = df.loc[idx, features_list].values.reshape(1, -1)
        prediction = model.predict(row_features)[0]
        recognized_classes.append(prediction)
        df.at[idx, 'primary_label'] = prediction

        progress_bar.progress((i + 1) / total)
        time.sleep(0.1)  # Simulate processing time

    st.success(f"Completed recognizing {total} missing primary_label values.")
    return df


def display_ebird_info(row):
    st.subheader("eBird Information")
    species = row['primary_label']
    ebird_url = f"https://api.ebird.org/v2/data/obs/{{species}}/recent"  # Replace with actual API URL

    try:
        response = requests.get(ebird_url)
        if response.status_code == 200:
            data = response.json()
            # Display sample image
            st.image(data["image_url"], caption=species)
            # Play audio if available
            if "audio_url" in data:
                st.audio(data["audio_url"])
        else:
            st.error(f"Failed to fetch data from eBird API. Status code: {response.status_code}")
    except Exception as e:
        st.error(f"Error fetching data from eBird API: {e}")


if uploaded_file:
    # Validate file structure
    data = validate_file(uploaded_file)

    if data is not None:
        # Check for 'primary_label' column
        if 'primary_label' not in data.columns:
            st.warning("Column 'primary_label' is missing. It will be predicted using the model.")
            data['primary_label'] = np.nan

        # Check for missing values in 'primary_label'
        if data['primary_label'].isna().any():
            if st.button("Recognize Missing Classes"):
                # Train a simple model for demonstration purposes
                st.info("Training model...")
                X = data.drop(columns=['primary_label']).dropna()
                y = data['primary_label'].dropna()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestClassifier()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                st.info(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

                # Recognize missing classes
                data = recognize_classes(data, model)

        # Display data
        st.subheader("Data Preview")
        if len(data) > 1000:
            st.dataframe(data.head(1000))
        else:
            st.dataframe(data)

        # Filtering
        filter_column = st.selectbox("Filter by column:", options=data.columns)
        filter_value = st.text_input("Enter value to filter by:")

        if filter_value:
            filtered_data = data[data[filter_column].astype(str).str.contains(filter_value)]
            st.dataframe(filtered_data)

        # Row selection
        selected_row = st.number_input("Select row to view eBird info:", min_value=0, max_value=len(data) - 1, step=1)

        if st.button("Show eBird Info"):
            display_ebird_info(data.iloc[selected_row])

        # Save updated data
        if st.button("Save as CSV"):
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Updated CSV", csv, "updated_data.csv")
