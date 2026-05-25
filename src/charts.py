import plotly.express as px
import streamlit as st


def show_value_distribution(df, column):

   # Show exact value counts using bar chart.

    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, "Count"]

    fig = px.bar(
        value_counts,
        x=column,
        y="Count",
        text="Count",
        title=f"Value Distribution of {column}",
        template="plotly_white"
    )

    # Show count labels above bars
    fig.update_traces(textposition="outside")

    st.plotly_chart(fig, use_container_width=True)
def show_scatter_plot(df, x_column, y_column):
    
    #Display scatter plot
    
    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        title=f"{x_column} vs {y_column}",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


def show_box_plot(df, column):
    
    #Display box plot

    fig = px.box(
        df,
        y=column,
        title=f"Box Plot of {column}",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


def show_correlation_bar_chart(df, target_column):
    
    #Show correlation of numeric features with the selected target column

    numeric_df = df.select_dtypes(include=["number"])

    if target_column not in numeric_df.columns:
        st.warning("Target column must be numeric for correlation chart.")
        return

    correlations = numeric_df.corr()[target_column].drop(target_column)
    correlations = correlations.sort_values(ascending=False).reset_index()
    correlations.columns = ["Feature", "Correlation"]

    fig = px.bar(
        correlations,
        x="Feature",
        y="Correlation",
        title=f"Feature Correlation with {target_column}",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)