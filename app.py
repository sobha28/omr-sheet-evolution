import streamlit as st
import numpy as np
import cv2
import os
import sqlite3
import json
import pandas as pd
from datetime import datetime
from omr_processor import process_omr_sheet

# --- DATABASE SETUP (using SQLite) ---
DB_FILE = "omr_results.db"

def init_db():
    """Initializes the database and creates the results table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            total_score REAL NOT NULL,
            subject_scores TEXT NOT NULL,
            processed_image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_result(total_score, subject_scores, processed_image_path):
    """Adds a new evaluation result to the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Convert subject_scores dict to a JSON string for storing
    subject_scores_json = json.dumps(subject_scores)
    
    c.execute("INSERT INTO results (timestamp, total_score, subject_scores, processed_image_path) VALUES (?, ?, ?, ?)",
              (timestamp, total_score, subject_scores_json, processed_image_path))
    conn.commit()
    conn.close()

def get_all_results():
    """Fetches all results from the database and returns as a DataFrame."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT timestamp, total_score FROM results ORDER BY timestamp DESC", conn)
    conn.close()
    return df

# --- Main function to process an image from any source ---
def handle_image_processing(image_buffer):
    """Takes an image buffer, processes it, and displays results."""
    bytes_data = image_buffer.getvalue()
    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    st.info("Image received. Processing...")

    with st.spinner('Finding bubbles and calculating score...'):
        try:
            temp_filename = f"static/uploads/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(temp_filename, cv_img)

            score, subject_scores, result_image_path = process_omr_sheet(temp_filename)
            add_result(score, subject_scores, result_image_path)

            st.success('Processing Complete!')
            st.balloons()

            st.header("📊 Evaluation Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Score", f"{score:.2f}%")
                st.write("Scores by Subject:")
                st.json(subject_scores)
            with col2:
                st.image(result_image_path, caption="Processed Sheet with Marks", use_column_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("Please try again. Ensure the entire sheet is visible and well-lit.")

# --- STREAMLIT APP LAYOUT ---
st.set_page_config(page_title="OMR Scoring App", page_icon="📝")
init_db()
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/processed", exist_ok=True)

st.title("📸 Automated OMR Scoring System")
st.write("Use your camera to scan an OMR sheet or upload an image file.")

# --- TAB LAYOUT FOR CAMERA AND UPLOAD ---
tab1, tab2 = st.tabs(["📷 Take Photo", "⬆️ Upload File"])

with tab1:
    st.subheader("Point your camera at the OMR sheet")
    camera_buffer = st.camera_input("Take a photo", label_visibility="collapsed")
    if camera_buffer:
        handle_image_processing(camera_buffer)

with tab2:
    st.subheader("Upload an image of the OMR sheet")
    upload_buffer = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    if upload_buffer:
        handle_image_processing(upload_buffer)

# --- DISPLAY DATABASE RESULTS ---
st.header("📜 Past Results")
results_df = get_all_results()

if results_df.empty:
    st.write("No results have been recorded yet.")
else:
    st.dataframe(results_df, use_container_width=True)