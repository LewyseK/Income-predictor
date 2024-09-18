import streamlit as st
import yaml
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import Hasher
import data, predict, history, dashboard

# Configure page
st.set_page_config(page_title="Income Predictor App", page_icon="🔮", layout="wide")

def main():
    # st.session_state.authentication_status = None

    # Load yaml configuration file
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Pre-hash all plain text passwords
    Hasher.hash_passwords(config["credentials"])

    authenticator = stauth.Authenticate(
                                        config["credentials"],
                                        config["cookie"]["name"],
                                        config["cookie"]["key"],
                                        config["cookie"]["expiry_days"],
                                        config["pre-authorized"],
                                        auto_hash = False
                                        )

    # Add authentication widget
    authenticator.login("sidebar", "login")

    # Set conditional statements
    if st.session_state["authentication_status"] is None:
        st.sidebar.info("Please enter username and password to access pages")
        st.sidebar.code(
                """
                Test Account's Credentials:
                Username: test_user
                Password: user123
                """
                )
        # Add image as interface
        st.image("assets/stacks-coins.jpg", width = 1000, use_column_width = True)


    if st.session_state["authentication_status"] is False:
        st.sidebar.error("Password or Username incorrect!! Please try again.")

    if st.session_state["authentication_status"]:
        if "page" not in st.session_state:
            st.session_state["page"] = "Home Page"    
        
        # Add logout button to side bar
        st.sidebar.info(f"### Welcome, *{st.session_state['name']}*!!")
        authenticator.logout(location = "sidebar")
        
        # Navigation menu
        st.sidebar.title("Navigation")
        st.session_state["page"] = st.sidebar.selectbox("## Please select a page here 👇", options = ["Home Page", "Data Page", "Predict Page", "History Page", "Dashboard Page"])

        # Show a page based on selection
        if st.session_state["page"] == "Home Page":
            show_home_page()
        elif st.session_state["page"] == "Data Page":
            data.show_data()
        elif st.session_state["page"] == "Predict Page":
            predict.show_predictions()
        elif st.session_state["page"] == "History Page":
            history.show_history()
        elif st.session_state["page"] == "Dashboard Page" :
            dashboard.show_dashboard()


# Create home page
def show_home_page():
    left, middle, right = st.columns([1, 10, 1])
    with middle:
        st.markdown("<h1 style='color: lightgreen;'> 🔮 INCOME LEVEL PREDICTION APP</h1>", unsafe_allow_html=True)
        
    st.markdown(
                """
                The app uses a machine learning model to predict whether an individual, based on specific characteristics,
                is likely to earn above or below a $50,000 threshold.
                """
                )

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.write("### 🤖 Machine Learning Integration")
        st.write(
                 """
                 The app uses a combination of advanced machine learning algorithms,
                 including XGBoost and Random Forest, to provide accurate predictions.
                 """
                 )
        
        left, right = st.columns(2)
        with left:
            st.markdown(
                        """
                        **XGBoost Classifier**
                        - `Prediction Accuracy: 94%`
                        - `Recall: 94%`
                        """
                        )
                        
        with right:
            st.markdown(
                        """
                        **Random Forest Classifier**
                        - `Prediction Accuracy: 95%`
                        - `Recall: 95%`
                        """
                        )
                        
            
        st.write("### 🧰 Key Features")
        st.write(
                    """
                    - **Data**: Displays both inbuilt and uploaded datasets.
                    - **Predict**: Displays customer churn status and prediction probability.
                    - **Dashboard**: Shows interactive data visualizations for quick insghts.
                    - **History**: Shows past predictions.
                    """
                    )
            
    with col2:                   
        st.write("### 👥 User Benefits")
        st.write(
                 """
                    - **Accurate Predictions**: Benefit from high precision and recall.
                    - **Informed Decisions**: Data-driven insights for better decision-making.
                    - **Real-Time Monitoring**: Predict income levels and visualize the results instantly.
                    - **User-Friendly**: A seamless and intuitive user experience.
                 """
                 )

                  
        with st.expander("**Need Help?**", expanded = False):
            st.markdown(
                        """
                        - Refer to [Documentation](https://github.com/LewyseK/Income-Predictor)
                        - <img src='https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/gmail.svg' alt='Email' height='20' style='filter: invert(100%) sepia(0%) saturate(0%) hue-rotate(0deg) brightness(100%) contrast(100%); margin-right: 8px;'> lewykk2011@gmail.com
                        """,
                        unsafe_allow_html = True
                        )

        with st.expander("#### 👨‍💻 About Developer", expanded = False):
            st.write(
                      """
                      A committed data and business analyst with expertise in Data Analytics and Machine Learning,
                      I utilize data-driven insights and advanced algorithms to address intricate business problems and guide strategic decision-making.
                      """
                      )

        with st.expander("**Developer's Portfolio**", expanded = True):
            st.markdown(
                      """
                    - <img src='https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/github.svg' alt='GitHub' height='20' style='filter: invert(100%) sepia(0%) saturate(0%) hue-rotate(0deg) brightness(100%) contrast(100%); margin-right: 8px;'> [GitHub](https://github.com/LewyseK)
                    - <img src='https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/linkedin.svg' alt='LinkedIn' height='20' style='filter: invert(100%) sepia(0%) saturate(0%) hue-rotate(0deg) brightness(100%) contrast(100%); margin-right: 8px;'> [LinkedIn](https://www.linkedin.com/in/Lewis-Kamau-Karanja)
                    - <img src='https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/medium.svg' alt='Medium' height='20' style='filter: invert(100%) sepia(0%) saturate(0%) hue-rotate(0deg) brightness(100%) contrast(100%); margin-right: 8px;'> [Medium](https://medium.com/@Lewykk2011)
                     """,
                     unsafe_allow_html = True
                     )

if __name__ == "__main__":
    main()