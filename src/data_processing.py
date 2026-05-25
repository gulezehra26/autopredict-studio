import pandas as pd


def clean_data(df):

    #This function cleans the uploaded dataset.
    #It handles missing values and removes duplicate rows.
    

    # Make a copy so original data is not changed
    cleaned_df = df.copy()

    # Count missing values before cleaning
    missing_before = cleaned_df.isnull().sum().sum()

    # Count duplicate rows before cleaning
    duplicates_before = cleaned_df.duplicated().sum()

    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()

    # Fill missing values
    for column in cleaned_df.columns:

        # If column is numeric, fill missing values with median
        if pd.api.types.is_numeric_dtype(cleaned_df[column]):
            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())

        # If column is text/category, fill missing values with mode
        else:
            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])

    # Count missing and duplicates after cleaning
    missing_after = cleaned_df.isnull().sum().sum()
    duplicates_after = cleaned_df.duplicated().sum()

    cleaning_report = {
        "missing_before": int(missing_before),
        "missing_after": int(missing_after),
        "duplicates_before": int(duplicates_before),
        "duplicates_after": int(duplicates_after),
        "rows_before": int(df.shape[0]),
        "rows_after": int(cleaned_df.shape[0])
    }

    return cleaned_df, cleaning_report