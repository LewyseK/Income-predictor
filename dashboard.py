import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os

def show_dashboard():
    # Radio widget to select dashboard choice
    dashboard_choice = st.sidebar.radio("Select Dashboard", ["EDA Dashboard", "KPI Dashboard"])

    # Load history data
    @st.cache_data
    def load_dashboard_data():
        if os.path.exists("data/uploaded_data_history.csv"):
            data = pd.read_csv("data/uploaded_data_history.csv")
        else:
            st.error("## No Data Available")
        return data
        
    df = load_dashboard_data()

    # Use the loaded data for dashboard based on user’s selected dashboard
    if df is not None and not df.empty:
    
        if dashboard_choice == "EDA Dashboard":

            # Align dashboad name to the centre
            left, middle, right = st.columns([1, 5, 1])
            with middle:
                # Sidebar for dashboard selection
                st.markdown("<h1 style='color: gold;'> 🔍 Exploratory Data Analysis</h1>", unsafe_allow_html=True)

            # Preview data frame
            st.dataframe(df.head())


            # Create 3 columns for layout
            col1, col2, col3 = st.columns(3)
            with col1:
                # Box plot for age, working_week_per_year, etc.
                fig1 = px.box(df[["age", "working_week_per_year", "industry_code", "occupation_code", "total_employed", "mig_year"]],
                            title = "Box plots of Key Features", template = "plotly_dark", color_discrete_sequence = ["red", "darkred"])
                st.plotly_chart(fig1)

            with col2:
                # Box plot for other important features
                fig2 = px.box(df[["losses", "importance_of_record", "gains", "wage_per_hour", "stocks_status"]],
                                title = "Box plots of Economic Factors", color_discrete_sequence = ["red", "darkred"])
                st.plotly_chart(fig2)

            with col3:
                # Gender vs Income
                fig4 = px.histogram(df, x = "gender", color = "income_above_limit", barmode = "stack",
                                    title = "Income Distribution by Gender", color_discrete_sequence = ["red", "darkred"])
                st.plotly_chart(fig4)


            # Align heatmap to the centre
            left, middle, right = st.columns([1, 10, 1])
            with middle:
                # Global Correlation Heatmap (applicable to both dashboards)
                filter_df = df[["age", "employment_stat", "wage_per_hour", "working_week_per_year", "industry_code", "occupation_code",
                                "total_employed", "vet_benefit", "gains", "losses", "stocks_status", "mig_year", "importance_of_record", "income_above_limit"]]

                # Dropping NaN values
                filter_df.dropna(inplace = True)

                # Mapping the target variable for correlation
                filter_df["income_above_limit"] = filter_df["income_above_limit"].map({"Above limit": 0, "Below limit": 1})

                # Compute the correlation matrix
                corr = filter_df.corr()

                # Heatmap Plot using Plotly
                heatmap_fig = go.Figure(data = go.Heatmap(z = corr.values,
                                                        x = corr.columns,
                                                        y = corr.columns,
                                                        colorscale = "Viridis",
                                                        text = corr.values,  # Add the correlation values as text
                                                        texttemplate = "%{text:.2f}",
                                                        showscale = False  # Format the text to 2 decimal places)
                                                        ))

                heatmap_fig.update_layout(title = "Correlation Heatmap", width = 800, height = 600)
                st.plotly_chart(heatmap_fig, use_container_width = True)


            # Create 3 columns for layout
            col1, col2 = st.columns(2)
            with col1:
                # Education vs Income
                fig5 = px.histogram(df, x = "education", color = "income_above_limit", barmode = "stack",
                                    title = "Income Distribution by Education", color_discrete_sequence = ["red", "darkred"])
                st.plotly_chart(fig5)

            with col2:
                # Marital Status vs Income
                fig6 = px.histogram(df, x = "marital_status", color = "income_above_limit", barmode = "stack",
                                    title = "Income Distribution by Marital Status", color_discrete_sequence = ["red", "darkred"])
                st.plotly_chart(fig6)


            # Align heatmap to the centre
            left, middle, right = st.columns([1, 10, 1])
            with middle:
                # Define the columns to be used
                selected_columns = ["employment_stat", "wage_per_hour", "mig_year", "importance_of_record", "income_above_limit"]

                # Create a scatter matrix
                fig = px.scatter_matrix(
                    df[selected_columns],
                    dimensions = selected_columns[:-1], 
                    color = "income_above_limit", 
                    title = "Pair Plot using Plotly",
                    labels = {col: col.replace("_", " ").capitalize() for col in selected_columns},  # Custom labels
                    color_discrete_sequence = ["red", "darkred"]
                )

                # Update the layout
                fig.update_layout(
                    width = 1000,
                    height = 800,
                    title_font_size = 18,
                    hoverlabel = dict(font_size = 12)
                )
                st.plotly_chart(fig, use_container_width = True)



        elif dashboard_choice == "KPI Dashboard":
            # Align title of dashboard to the centre
            left, middle, right = st.columns([1, 5, 1])
            with middle:
                st.markdown("<h1 style='color: gold;'> 💡 Income Level Indicator Dashboard</h1>", unsafe_allow_html=True)

            # Sidebar widgets
            st.sidebar.header("Filter Options")

            # Age slider
            age = st.sidebar.slider("Age", 0, int(df["age"].max()), (0, int(df["age"].max())), key = "age")

            # Gender selectbox
            gender = st.sidebar.selectbox("Gender", options = ["Female", "Male"], key = "gender")

            # Tax status selectbox
            tax_status = st.sidebar.selectbox("Tax Status", options = [
                "Head of household", "Single", "Nonfiler", "Joint both 65+", "Joint both under 65", "Joint one under 65 & one 65+"], key = "tax_status")

            # Filter the data based on sidebar input
            filtered_df = df[(df["age"] >= age[0]) & (df["age"] <= age[1]) & 
                            (df["gender"] == gender) & 
                            (df["tax_status"] == tax_status)]


            # Check if the filtered DataFrame is empty
            if filtered_df.empty:
                st.warning("No data available for the selected filters. Please adjust your filters.")
            else:
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Above Limit Count", filtered_df[filtered_df["income_above_limit"] == "Above limit"].shape[0])

                with col2:
                    st.metric("Below Limit Count", filtered_df[filtered_df["income_above_limit"] == "Below limit"].shape[0])

                with col3:
                    st.metric("Male Count", filtered_df[filtered_df["gender"] == "Male"].shape[0])

                with col4:
                    st.metric("Female Count", filtered_df[filtered_df["gender"] == "Female"].shape[0])
                    

                # Split layout into two columns for visualizations -- Row 1
                left, right = st.columns(2)
                with left:
                    # Create Pie chart
                    income_counts = filtered_df["income_above_limit"].value_counts()
                    pie_fig = px.pie(values = income_counts, names = income_counts.index,
                                    title = "Percentage Count of Income Limit",
                                    color_discrete_sequence = ["blue", "lightblue"],
                                    hole = 0.45,
                                    width = 650,
                                    height = 400)
                    
                    st.plotly_chart(pie_fig)

                with right:
                    # Plotly Express Choropleth for country of birth
                    fig_map = px.choropleth(
                        filtered_df, 
                        locations = "country_of_birth_own", 
                        locationmode = "country names",
                        color = "income_above_limit", 
                        hover_name = "country_of_birth_own", 
                        color_continuous_scale = px.colors.sequential.Plasma,
                        title = "Income Disparity by Country of Birth",
                        width = 650,
                        height = 400)
                    
                    # Display the map in Streamlit
                    st.plotly_chart(fig_map)


                # Split layout into two columns for visualizations -- Row 1
                left, right = st.columns(2)      
                with left:
                    # Create stacked histogram chart
                    fig = px.histogram(filtered_df, 
                                    x = "education", 
                                    color = "income_above_limit", 
                                    barmode = "stack", 
                                    labels = {"education": "Education", "count": "Count"},
                                    color_discrete_sequence = ["blue", "lightblue"],
                                    height = 450)

                    # Update layout
                    fig.update_layout(yaxis_title = "Education", 
                                    xaxis_title = "Count", 
                                    title = "Count of Education Levels by Income Above Limit")
                    # Show the plot
                    st.plotly_chart(fig)

                with right:
                    # Create stacked histogram chart
                    fig = px.histogram(filtered_df, 
                                    x = "industry_code_main", 
                                    color = "income_above_limit", 
                                    barmode = "stack", 
                                    labels = {"industry_code_main": "Industry", "count": "Count"},
                                    color_discrete_sequence = ["blue", "lightblue"],
                                    height = 450)

                    # Update layout
                    fig.update_layout(yaxis_title = "Industry", 
                                    xaxis_title = "Count", 
                                    title = "Count of Industry Codes by Income Above Limit")
 
                    # Show the plot
                    st.plotly_chart(fig)





                

























































































































