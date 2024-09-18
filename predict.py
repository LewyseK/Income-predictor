import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import os

def show_predictions():
    # Load XGBoost model and threshold
    @st.cache_resource(show_spinner = "XGBoost Model Loading")
    def load_xgboost_model():
        model = joblib.load("models/XGBoost.joblib")
        return model

    # Load Random Forest model and threshold
    @st.cache_resource(show_spinner = "Random Forest Model Loading")
    def load_random_forest_model():
        model = joblib.load("models/RandomForest.joblib")
        return model

    def select_model():
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h1 style='color: gold;'> 🔮 Prediction Hub</h1>", unsafe_allow_html=True)
        with col2:
            st.selectbox("Select a model", options=["XGBoost", "Random Forest"], key = "selected_model")

        if st.session_state["selected_model"] == "XGBoost":
            pipeline = load_xgboost_model()
        else:
            pipeline = load_random_forest_model()

        if pipeline:
            encoder = joblib.load("models/Encoder.joblib")
        else:
            encoder = None

        return pipeline, encoder

    # Initialize session state
    if "probability" not in st.session_state:
        st.session_state["probability"] = None
    if "prediction" not in st.session_state:
        st.session_state["prediction"] = None

    # Function to make a single prediction
    def make_single_prediction(pipeline, encoder):
        data = [[st.session_state["age"], st.session_state["gender"], st.session_state["education"],
                st.session_state["marital_status"], st.session_state["race"], st.session_state["is_hispanic"],
                st.session_state["employment_commitment"], st.session_state["employment_stat"], st.session_state["wage_per_hour"],
                st.session_state["working_week_per_year"], st.session_state["industry_code"], st.session_state["industry_code_main"],
                st.session_state["occupation_code"], st.session_state["total_employed"], st.session_state["household_stat"],
                st.session_state["household_summary"], st.session_state["vet_benefit"], st.session_state["tax_status"],
                st.session_state["gains"], st.session_state["losses"], st.session_state["stocks_status"], st.session_state["citizenship"],
                st.session_state["mig_year"], st.session_state["country_of_birth_own"], st.session_state["country_of_birth_father"],
                st.session_state["country_of_birth_mother"], st.session_state["importance_of_record"]]]

        columns = ["age", "gender", "education", "marital_status", "race", "is_hispanic", "employment_commitment",
                "employment_stat", "wage_per_hour", "working_week_per_year", "industry_code", "industry_code_main",
                "occupation_code", "total_employed", "household_stat", "household_summary", "vet_benefit", "tax_status",
                "gains", "losses", "stocks_status", "citizenship", "mig_year", "country_of_birth_own", "country_of_birth_father",
                "country_of_birth_mother", "importance_of_record"]

        df = pd.DataFrame(data, columns=columns)

        probability = pipeline.predict_proba(df)
        pred = (probability[:, 1] >= 0.5).astype(int)
        pred = int(pred[0])
        prediction = encoder.inverse_transform([pred])[0]

        # Save the prediction history
        now = datetime.datetime.now()
        history_df = df.copy()
        history_df.insert(0, "Prediction_Date", now.date())
        history_df.insert(1, "Prediction_Time", now.strftime("%H:%M"))
        history_df["Model_used"] = st.session_state["selected_model"]
        history_df["income_above_limit"] = prediction
        history_df["Probability"] = np.where(pred == 0, np.round(probability[:, 0] * 100, 2), np.round(probability[:, 1] * 100, 2))
        history_df.to_csv("./data/history.csv", mode = "a", header = not os.path.exists("./data/history.csv"), index = False)

        st.session_state["probability"] = probability
        st.session_state["prediction"] = prediction

        return probability, prediction
    

    # Single prediction entry form
    def entry_form(pipeline, encoder):
        st.markdown("#### Enter Information")
        with st.form(key = "Individual info"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.selectbox("Gender", options = ["Female", "Male"], key = "gender")

                st.number_input("Age", min_value = 0, key = "age")

                st.selectbox("Citizenship", options = [
                    "Native", "Foreign born- Not a citizen of U S", "Foreign born- U S citizen by naturalization", "Native- Born abroad of American Parent(s)",
                    "Native- Born in Puerto Rico or U S Outlying"], key = "citizenship")
               
                st.selectbox("Marital Status", options = [
                    "Widowed", "Never married", "Married-civilian spouse present", "Divorced", "Married-spouse absent",
                    "Separated", "Married-A F spouse present"], key = "marital_status")
                
                st.selectbox("Race", options = [
                    "White", "Black", "Asian or Pacific Islander", "Amer Indian Aleut or Eskimo", "Other"], key = "race")
                
                st.selectbox("Hispanic ethnicity", options = [
                    "All other", "Mexican-American", "Central or South American", "Mexican (Mexicano)", "Puerto Rican", "Other Spanish",
                    "Cuban", "Do not know", "Chicano"], key = "is_hispanic")
                
                st.selectbox("Employment Commitment", options = [
                    "Not in labor force", "Children or Armed Forces", "Full-time schedules", "PT for econ reasons usually PT",
                    "Unemployed full-time", "PT for non-econ reasons usually FT", "PT for econ reasons usually FT", "Unemployed part- time"], key = "employment_commitment")
                                
            with col2:
                st.number_input("Employment Status", min_value = 0, max_value = 2, key = "employment_stat")

                st.selectbox("Education Level", options = [
                    "High school graduate", "12th grade no diploma", "Children", "Bachelors degree(BA AB BS)", "7th and 8th grade",
                    "11th grade", "9th grade", "Masters degree(MA MS MEng MEd MSW MBA)", "10th grade", "Associates degree-academic program",
                    "1st 2nd 3rd or 4th grade", "Some college but no degree", "Less than 1st grade", "Associates degree-occup /vocational",
                    "Prof school degree (MD DDS DVM LLB JD)", "5th or 6th grade", "Doctorate degree(PhD EdD)"], key = "education")

                st.selectbox("Household Summary", options = [
                    "Householder", "Child 18 or older", "Child under 18 never married", "Spouse of householder", "Nonrelative of householder",
                    "Other relative of householder", "Group Quarters- Secondary individual", "Child under 18 ever married"], key = "household_summary")
               
                st.selectbox("Industry Code Main", options = [
                    "Not in universe or children", "Hospital services", "Retail trade", "Finance insurance and real estate", "Manufacturing-nondurable goods",
                    "Transportation", "Business and repair services", "Medical except hospital", "Education","Construction",
                    "Manufacturing-durable goods", "Public administration", "Agriculture", "Other professional services", "Mining",
                    "Utilities and sanitary services", "Private household services", "Personal services except private HH",
                    "Wholesale trade", "Communications", "Entertainment", "Social services", "Forestry and fisheries", "Armed Forces"], key = "industry_code_main")
                
                st.selectbox("Country of Birth", options = [
                    "US", "El-Salvador", "Mexico", "Philippines", "Cambodia", "China", "Hungary", "Puerto-Rico", "England", "Dominican-Republic",
                    "Japan", "Canada", "Ecuador", "Italy", "Cuba", "Peru", "Taiwan", "South Korea", "Poland", "Nicaragua", "Germany", "Guatemala",
                    "India", "Ireland", "Honduras", "France", "Trinadad&Tobago", "Thailand", "Iran", "Vietnam", "Portugal", "Laos", "Panama",
                    "Scotland", "Columbia", "Jamaica", "Greece", "Haiti", "Yugoslavia", "Outlying-U S (Guam USVI etc)", "Holand-Netherlands",
                    "Hong Kong"], key = "country_of_birth_own")
                
                st.selectbox("Father's Country of Birth", options = [
                    "US", "India", "Poland", "Germany", "El-Salvador", "Mexico", "Puerto-Rico", "Philippines", "Greece", "Canada", "Ireland",
                    "Cambodia", "Ecuador", "China", "Hungary", "Dominican-Republic", "Japan", "Italy", "Cuba", "Peru", "Jamaica",
                    "South Korea", "Yugoslavia", "Nicaragua", "Columbia", "Guatemala", "France", "England", "Iran", "Honduras", "Haiti",
                    "Trinadad&Tobago", "Outlying-U S (Guam USVI etc)", "Thailand", "Vietnam", "Hong Kong", "Portugal", "Laos", "Scotland", "Taiwan",
                    "Holand-Netherlands", "Panama"], key = "country_of_birth_father")
                
                st.selectbox("Mother's Country of Birth", options = [
                    "US", "India", "Peru", "Germany", "El-Salvador", "Mexico", "Puerto-Rico", "Philippines", "Canada", "France", "Cambodia",
                    "Italy", "Ecuador", "China", "Hungary", "Dominican-Republic", "Japan", "England", "Cuba", "Poland", "South Korea", "Yugoslavia",
                    "Scotland", "Nicaragua", "Guatemala", "Holand-Netherlands", "Greece", "Ireland", "Honduras", "Haiti", "Outlying-U S (Guam USVI etc)",
                    "Trinadad&Tobago", "Thailand", "Jamaica", "Iran", "Vietnam", "Columbia", "Portugal", "Laos", "Taiwan", "Hong Kong", "Panama"], key = "country_of_birth_mother")
                                       
            with col3:
                st.selectbox("Household Status", options = [
                    "Householder", "Nonfamily householder", "Child 18+ never marr Not in a subfamily", "Child <18 never marr not in subfamily",
                    "Spouse of householder", "Child 18+ spouse of subfamily RP", "Secondary individual", "Child 18+ never marr RP of subfamily",
                    "Other Rel 18+ spouse of subfamily RP", "Grandchild <18 never marr not in subfamily", "Other Rel <18 never marr child of subfamily RP",
                    "Other Rel 18+ ever marr RP of subfamily", "Other Rel 18+ ever marr not in subfamily", "Child 18+ ever marr Not in a subfamily", "RP of unrelated subfamily",
                    "Child 18+ ever marr RP of subfamily", "Other Rel 18+ never marr not in subfamily", "Child under 18 of RP of unrel subfamily",
                    "Grandchild <18 never marr child of subfamily RP", "Grandchild 18+ never marr not in subfamily", "Other Rel <18 never marr not in subfamily",
                    "In group quarters", "Grandchild 18+ ever marr not in subfamily", "Other Rel 18+ never marr RP of subfamily", "Child <18 never marr RP of subfamily",
                    "Grandchild 18+ never marr RP of subfamily", "Spouse of RP of unrelated subfamily", "Grandchild 18+ ever marr RP of subfamily",
                    "Child <18 ever marr not in subfamily", "Child <18 ever marr RP of subfamily", "Other Rel <18 ever marr RP of subfamily",
                    "Grandchild 18+ spouse of subfamily RP", "Child <18 spouse of subfamily RP", "Other Rel <18 ever marr not in subfamily", "Other Rel <18 never married RP of subfamily",
                    "Other Rel <18 spouse of subfamily RP", "Grandchild <18 ever marr not in subfamily", "Grandchild <18 never marr RP of subfamily"], key = "household_stat")
                
                st.selectbox("Tax Status", options = [
                    "Head of household", "Single", "Nonfiler", "Joint both 65+", "Joint both under 65", "Joint one under 65 & one 65+"], key = "tax_status")
                
                st.number_input("Wage per Hour", min_value = 0, key = "wage_per_hour")

                st.number_input("Working Week per Year", min_value = 0, key = "working_week_per_year")

                st.number_input("Industry Code", min_value = 0, key = "industry_code")

                st.number_input("Occupation Code", min_value = 0, key = "occupation_code")

                st.number_input("Total Employed", min_value = 0, key = "total_employed")

            with col4:
                st.number_input("Vet Benefit", min_value = 0, max_value = 2, key = "vet_benefit")

                st.number_input("Gains", min_value = 0, key = "gains")

                st.number_input("Losses", min_value = 0, key = "losses")

                st.number_input("Stocks Status", min_value = 0, key = "stocks_status")

                st.number_input("Migration Year", min_value = 94, max_value = 95, key = "mig_year")

                st.number_input("Importance of Record", min_value = 0.00, key = "importance_of_record")               

            submit_button = st.form_submit_button("Make Prediction")

            if submit_button:
                make_single_prediction(pipeline, encoder)


    # Bulk prediction function
    def bulk_prediction(model, df, encoder):
        prob_score = model.predict_proba(df.drop(columns=["ID"]))
        bulk_pred = (prob_score[:, 1] >= 0.5).astype(int)
        bulk_prediction = encoder.inverse_transform(bulk_pred)
        return bulk_prediction, prob_score

    # Function for bulk prediction
    def make_bulk_prediction(df, is_uploaded_data, encoder):
        expected_features = [
            "ID", "age", "gender", "education", "marital_status", "race", "is_hispanic", "employment_commitment", "employment_stat", "wage_per_hour",
            "working_week_per_year", "industry_code", "industry_code_main", "occupation_code", "total_employed", "household_stat",
            "household_summary", "vet_benefit", "tax_status", "gains", "losses", "stocks_status", "citizenship", "mig_year", "country_of_birth_own",
            "country_of_birth_father", "country_of_birth_mother", "importance_of_record"
        ]

        # Option to use previously uploaded data
        if st.checkbox("Use previously uploaded data"):
            if "uploaded_data" in st.session_state:
                df = st.session_state["uploaded_data"]
                st.info("Using previously uploaded data for bulk prediction:")
                st.dataframe(df.head())
            else:
                st.warning("No data found in the session. Please upload a file first.")

        if st.button("Make Bulk Prediction"):
            if all(feature in df.columns for feature in expected_features):
                # Load model
                model = load_xgboost_model()

                if model is not None:
                    bulk_predict, probability_score = bulk_prediction(model, df, encoder)

                    # Add relevant information and save to file
                    bulk_history_df = df.copy()
                    now = datetime.datetime.now()
                    bulk_history_df.insert(1, "Prediction_Date", now.date())
                    bulk_history_df["Model_used"] = "Gradient Boost Classifier"
                    bulk_history_df["income_above_limit"] = bulk_predict
                    bulk_history_df["Probability"] = np.where(bulk_predict == 0, np.round(probability_score[:, 0] * 100, 2), np.round(probability_score[:, 1] * 100, 2))
                    history_file = "./data/uploaded_data_history.csv" if is_uploaded_data else "./data/inbuilt_data_history.csv"
                    bulk_history_df.to_csv(history_file, mode = "w", header = True, index = False)

                    st.success("Bulk Predictions made successfully.")
                else:
                    st.error("### Failed to load XGBoost Model!!")
            else:
                st.error("Uploaded data does not match expected features.")

        # Button to preview prediction history
        if st.button("Preview Prediction"):
            history_file = "./data/uploaded_data_history.csv" if is_uploaded_data else "./data/inbuilt_data_history.csv"
            if os.path.exists(history_file):
                history_df = pd.read_csv(history_file)
                st.dataframe(history_df.head())
            else:
                st.warning("### No prediction history found")

        return 

    # Main function
    def main():
        # Radio button for prediction type
        prediction_type = st.sidebar.radio("Choose Prediction Type", ["Single Prediction", "Bulk Prediction"])

        pipeline, encoder = select_model()

        if prediction_type == "Single Prediction":
            entry_form(pipeline, encoder)

            prediction = st.session_state["prediction"]
            probability = st.session_state["probability"]

            st.sidebar.info("### View Prediction Result Here 👇")

            if prediction == "Below limit":
                st.sidebar.warning(f"### The individual is likely to receive an income below the threshold.\nProbability: {probability[0][1] * 100:.2f}%")
            elif prediction == "Above limit":
                st.sidebar.success(f"### The individual is unlikely to receive an income above the threshold.\nProbability: {probability[0][0] * 100:.2f}%")

        elif prediction_type == "Bulk Prediction":
            # Assume the uploaded data is stored in session state
            if "uploaded_data" in st.session_state:
                df = st.session_state["uploaded_data"]
                make_bulk_prediction(df, True, encoder)
            else:
                st.warning("Please upload data on the Data Page for bulk predictions.")

    main()