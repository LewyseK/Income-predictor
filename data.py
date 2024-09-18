import streamlit as st
import pandas as pd
import numpy as np
import time

def show_data():
    user_choice = st.sidebar.radio("### Select a page", options = ["Data Understanding", "Data Hub"], key = "option_selected")

    if user_choice == "Data Understanding":
        st.info("## 📚 Data Understanding")
        st.markdown(
                        """
                        The dataset consists of various features that provide detailed information about individuals,
                        ranging from demographic details to employment and migration history. The models were trained
                        on these features

                        #### Feature Description:
                        """
                        )

        # Create columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                - **ID**: Unique identifier for each individual.
                - **age**: Age of the individual.
                - **gender**: Gender of the individual.
                - **education**: Level of education of the individual.
                - **marital_status**: Marital status of the individual.
                - **race**: Race of the individual.
                - **is_hispanic**: Indicator for Hispanic ethnicity.
                - **employment_commitment**: Employment status of the individual.
                - **employment_stat**: Employment status of the individual **(0: Unemployed, 1: Employed, 2: Part-time)**
                - **wage_per_hour**: Hourly wage of the individual.
                - **working_week_per_year**: Number of weeks worked per year.
                - **industry_code**: Code representing the industry of employment.
                - **occupation_code**: Code representing the occupation.
                - **total_employed**: Total number of individuals employed.
                """
                )

        with col2:
            st.markdown(
                """
                - **vet_benefit**: Veteran benefits received.
                - **tax_status**: Tax status of the individual.
                - **gains**: Financial gains.
                - **losses**: Financial losses.
                - **stocks_status**: Status of stocks owned.
                - **citizenship**: Citizenship status.
                - **mig_year**: Year of migration.
                - **country_of_birth_own**: Country of birth of the individual.
                - **country_of_birth_father**: Country of birth of the individual’s father.
                - **country_of_birth_mother**: Country of birth of the individual’s mother.
                - **importance_of_record**: Importance of the record.
                - **income_above_limit**: Indicator for income above a certain limit.
                - **household_stat**: Household status.
                - **household_summary**: Summary of household information.
                """
                )


    if user_choice == "Data Hub":
        st.markdown("<h1 style='color: lightblue;'> 📚 Data Hub</h1>", unsafe_allow_html=True)
        st.info(f"### *Welcome To Data Hub*")
        st.markdown(f"#### **Uploaded data will be availbale for prediction in the predict page**") 

        with st.expander("## **Explore the dataset used for testing here**", expanded = False, icon = "👇"):
                # Load data
                def load_data():
                    df = pd.read_csv("data/Test.csv")
                    return df
                
                data = load_data()

                # Clean columns
                def clean_columns(data):
                    # List of columns to drop
                    columns_to_drop = [
                                "class", "education_institute", "unemployment_reason", "is_labor_union", 
                                "occupation_code_main", "under_18_family", "veterans_admin_questionnaire", 
                                "migration_code_change_in_msa", "migration_prev_sunbelt", 
                                "migration_code_move_within_reg", "migration_code_change_in_reg", 
                                "residence_1_year_ago", "old_residence_reg", "old_residence_state"]
                            
                    # Drop the specified columns
                    data = data.drop(columns = columns_to_drop, axis = 1)

                    # Iterate through the columns with category or object (string) data types
                    for col in data.select_dtypes(include = ["category", "object"]).columns:
                        data[col] = data[col].str.strip()

                    # Replace "?" with np.nan in the entire DataFrame
                    data.replace("?", np.nan, inplace = True)

                    # Replace "NA" with "All other" in the "is_hispanic" column
                    data["is_hispanic"] = data["is_hispanic"].replace("NA", "All other")
                            
                    return data
                
                # Display data preview
                st.data_editor(data)
                    
                # Display data statistics
                left, right = st.columns(2)
                with left:
                    st.write(f"#### Data Statistics")
                    st.write(data.describe())
                    st.write(f"#### Data Shape")
                    st.write(data.shape)
                    st.write(f"#### Number of Duplicated Rows")
                    st.write(data.duplicated().sum())

                with right:            
                    st.write(f"#### Missing Values")
                    st.write(data.isnull().sum())
                    st.write(f"#### Unique Values")
                    st.write(data.nunique())
                    

        with st.expander("**Upload your dataset here**", expanded = False, icon = "👇"):
                
                # Upload file
                upload_file = st.file_uploader(label="Choose a file", type=["csv", "xlsx"])

                # Initialize session state variable for tracking data load status
                if "data_loaded" not in st.session_state:
                    st.session_state.data_loaded = False

                if upload_file is not None:
                    # Check for file type and read file
                    try:
                        if upload_file.name.endswith(".csv"):
                            data = pd.read_csv(upload_file)
                        elif upload_file.name.endswith(".xlsx"):
                            data = pd.read_excel(upload_file)
                        else:
                            st.error("Unsupported file type!")
                            data = None
                    except Exception as e:
                        st.error(f"Error reading this file: {str(e)}")
                        data = None
                    
                    if data is not None:                  
                    
                        # Clean the columns
                        data = clean_columns(data)

                        # Store the data in session_state
                        st.session_state["uploaded_data"] = data
                        
                    # Display success message
                    if data is not None and not st.session_state.data_loaded:
                        success_msg = st.empty()
                        success_msg.success("Data uploaded successfully!")
                        # Display success message for 2 seconds
                        time.sleep(2)
                        # Clear success message
                        success_msg.empty()
                        st.session_state.data_loaded = True

                    if data is not None:
                        # Preview data
                        st.subheader("Data Preview")
                        st.write("First few rows of your data:")
                        st.dataframe(data.head())

                    # Option to display entire data
                    if st.checkbox("Show entire data (Optional)"):
                        st.dataframe(data)

                    # Option to explore data
                    if st.checkbox("Explore data (Optional)"):
                        # Display data statistics
                        left, right = st.columns(2)
                        with left:
                            st.write(f"#### Data Statistics")
                            st.write(data.describe())
                            st.write(f"#### Data Shape")
                            st.write(data.shape)
                            st.write(f"#### Number of Duplicated Rows")
                            st.write(data.duplicated().sum())

                        with right:            
                            st.write(f"#### Missing Values")
                            st.write(data.isnull().sum())
                            st.write(f"#### Unique Values")
                            st.write(data.nunique())
                    
                else:
                    st.info("Please upload a file to preview the data.")