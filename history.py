import streamlit as st
import pandas as pd
import os

def show_history():
    # History of Single predictions
    @st.cache_data
    def data_history():
        if os.path.exists("data/history.csv"):
            history_df = pd.read_csv("data/history.csv")
        else:
            history_df = pd.DataFrame()
        return history_df

    # Predictions history on uploaded data
    @st.cache_data
    def load_uploaded_data_history():
        if os.path.exists("data/uploaded_data_history.csv"):
            uploaded_data_history_df = pd.read_csv("data/uploaded_data_history.csv")
        else:
            uploaded_data_history_df = pd.DataFrame()
        return uploaded_data_history_df
        

    # View prediction history based on user's choice
    def view_prediction_history():
        user_choice = st.sidebar.radio("### Display Prediction History",
                                    options = ["Single Prediction", "Bulk Prediction (For uploaded data)"], key = "user_choice")
        df = None
        
        # Display the chosen data history
        if user_choice == "Single Prediction":
            st.info("### 🔓 Income Above Limit Unlocked")
            st.subheader("Single Prediction History")
            if st.button("View History"):
                df = st.dataframe(data_history().iloc[::-1])
                
        elif user_choice == "Bulk Prediction (For uploaded data)":
            st.info("### 🔓 Income Above Limit Unlocked")
            st.subheader("Bulk Prediction History (For Uploaded Data)")
            if st.button("View History"):
                # Load the historical data
                df = load_uploaded_data_history()

                if df is not None:
                    # Display data in streamlit
                    st.dataframe(df)
                    # Save df to session state
                    st.session_state["dashboard_data"] = df

        return df

    # Execute function
    view_prediction_history()