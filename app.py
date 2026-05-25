import streamlit as st
import pandas as pd
from src.model_training import train_regression_models
from src.data_processing import clean_data

from src.charts import (
    show_value_distribution,
    show_scatter_plot,
    show_box_plot,
    show_correlation_bar_chart
)

# Page settings
st.set_page_config(
    page_title="AutoPredict Studio",
    page_icon="📊",
    layout="wide"
)

# App title
st.title("📊 AutoPredict Studio")
st.subheader("Automated Machine Learning & Predictive Analytics Platform")

st.write(
    "Upload your dataset, clean it automatically, train regression models, "
    "visualize results, and make predictions."
)

st.divider()

# Dataset Upload Section

st.header("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV or Excel file",
    type=["csv","xlsx"]
)

if uploaded_file is not None:

    try:
        # Read CSV or Excel file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Save dataset in session memory
        st.session_state["df"] = df

        st.success("Dataset uploaded successfully!")

        # Dataset preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Dataset basic information
        st.subheader("Dataset Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric("Missing Values", int(df.isnull().sum().sum()))

        # Column information
        st.subheader("Column Information")

        column_info = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(column_info, use_container_width=True)
     # Show error if file cannot be read
    except Exception as e:
        st.error("Something went wrong while reading the file.")
        st.write(e)
# Ask if no file is selected
else:
    st.info("Please upload a CSV or Excel dataset to continue.")


# Automatic Data Cleaning Section

if "df" in st.session_state:

    st.divider()
    st.header("2. Automatic Data Cleaning")

    if st.button("Clean Dataset"):

        cleaned_df, cleaning_report = clean_data(st.session_state["df"])

        # Save cleaned data for later ML steps
        st.session_state["cleaned_df"] = cleaned_df

        st.success("Dataset cleaned successfully!")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Missing Before", cleaning_report["missing_before"])

        with col2:
            st.metric("Missing After", cleaning_report["missing_after"])

        with col3:
            st.metric("Duplicates Before", cleaning_report["duplicates_before"])

        with col4:
            st.metric("Duplicates After", cleaning_report["duplicates_after"])

        st.subheader("Cleaned Dataset Preview")
        st.dataframe(cleaned_df.head(10), use_container_width=True)


# Visualization Dashboard

if "cleaned_df" in st.session_state:

    st.divider()
    st.header("3. Visualization Dashboard")

    cleaned_df = st.session_state["cleaned_df"]

    numeric_columns = cleaned_df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_columns) > 0:

        # Histogram
        st.subheader("Value Distribution")

        distribution_column = st.selectbox(
            "Select column for value distribution",
            cleaned_df.columns.tolist()
        )

        show_value_distribution(cleaned_df, distribution_column)
        # Scatter plot
        st.subheader("Scatter Plot")

        col1, col2 = st.columns(2)

        with col1:
            scatter_x = st.selectbox(
                "Select X-axis",
                numeric_columns,
                key="scatter_x"
            )

        with col2:
            scatter_y = st.selectbox(
                "Select Y-axis",
                numeric_columns,
                key="scatter_y"
            )

        show_scatter_plot(cleaned_df, scatter_x, scatter_y)

        # Box plot
        st.subheader("Box Plot")

        box_column = st.selectbox(
            "Select column for box plot",
            numeric_columns,
            key="box_plot"
        )

        show_box_plot(cleaned_df, box_column)


        # Correlation chart
        st.subheader("Feature Correlation")

        target_column = st.selectbox(
            "Select target column for correlation analysis",
            numeric_columns,
            key="correlation_target"
        )

        show_correlation_bar_chart(cleaned_df, target_column)
    else: 
        st.warning("No numeric columns found for visualization.") 

# Machine Learning Training

if "cleaned_df" in st.session_state:
    # Add section title

    st.divider()
    st.header("4. Machine Learning Training")
    # Load cleaned dataset

    cleaned_df = st.session_state["cleaned_df"]
    # Select only numeric columns for target prediction

    numeric_columns = cleaned_df.select_dtypes(include=["number"]).columns.tolist()
    # Let user select target column

    target_column = st.selectbox(
        "Select target column for prediction",
        numeric_columns,
        key="ml_target_column"
    )
    # Start model training when button is clicked

    if st.button("Train Regression Models"):
        # Train models and get results

        results_df, best_model_name, best_score, feature_names = train_regression_models(
            cleaned_df,
            target_column
        )
        # Save feature names

        st.session_state["feature_names"] = feature_names
        # Show success message

        st.success("Models trained successfully!")

        # Results table
        st.subheader("Model Performance Comparison")
        st.dataframe(results_df, use_container_width=True)

        # Best model
        st.subheader("Best Model Selected")

        col1, col2 = st.columns(2)

        with col1:
            st.success(best_model_name)

        with col2:
            st.metric("Best R² Score", round(best_score, 4))

        # Comparison chart
        st.subheader("R² Score Comparison")

        comparison_df = results_df.sort_values("R2 Score", ascending=False)

        st.bar_chart(
            comparison_df.set_index("Model")["R2 Score"]
        )

        st.info(
            "AutoPredict Studio automatically selected the best model "
            "based on highest R² score."
        )
        # Convert results into downloadable CSV file
        csv_results = results_df.to_csv(index=False)

        st.download_button(
            label="Download Model Results",
            data=csv_results,
            file_name="model_results.csv",
            mime="text/csv"
        )