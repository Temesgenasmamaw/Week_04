import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore

def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # dtype of missing values
    mis_val_dtype = df.dtypes

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Dtype'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

def missing_data_summary(self) -> pd.DataFrame:
    """
    Returns a summary of columns with missing data, including count and percentage of missing values.
    Returns:
        pd.DataFrame: A DataFrame with columns 'Missing Count' and 'Percentage (%)' for columns with missing values.
    """
    # Total missing values per column
    missing_data = self.data.isnull().sum()
    
    # Filter only columns with missing values greater than 0
    missing_data = missing_data[missing_data > 0]
    
    # Calculate the percentage of missing data
    missing_percentage = (missing_data / len(self.data)) * 100
    
    # Combine the counts and percentages into a DataFrame
    missing_df = pd.DataFrame({
        'Missing Count': missing_data, 
        'Percentage (%)': missing_percentage
    })
    
    # Sort by percentage of missing data
    missing_df = missing_df.sort_values(by='Percentage (%)', ascending=False)
    
    return missing_df
    
def handle_missing_data(self, missing_type: str, missing_cols: list) -> pd.DataFrame:
    
    """
    Handles missing data based on predefined strategies.
    """
    if missing_type == 'high':
        # Drop columns with high missing data
        self.data = self.data.drop(columns=missing_cols, errors='ignore')
    elif missing_type == 'moderate':
        # Impute or drop columns with moderate missing data
        for col in missing_cols:
            if col in self.data.columns:
                if self.data[col].dtype == 'object':
                    # Impute categorical columns with mode (check if mode exists)
                    if not self.data[col].mode().empty:
                        self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
                    else:
                        self.data[col] = self.data[col].fillna('Unknown')  # Default for empty mode
                else:
                    # Impute numerical columns with median (check if median exists)
                    if not self.data[col].isnull().all():  # Ensure column has some numeric values
                        self.data[col] = self.data[col].fillna(self.data[col].median())
                    else:
                        self.data[col] = self.data[col].fillna(0)  # Default for empty median
    else:
        # Handle low missing data (default)
        for col in missing_cols:
            if col in self.data.columns:
                if self.data[col].dtype == 'object':
                    if not self.data[col].mode().empty:
                        self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
                    else:
                        self.data[col] = self.data[col].fillna('Unknown')  # Default for empty mode
                else:
                    if not self.data[col].isnull().all():
                        self.data[col] = self.data[col].fillna(self.data[col].median())
                    else:
                        self.data[col] = self.data[col].fillna(0)  # Default for empty median
    return self.data

def remove_all_columns_outliers(df, method="iqr"):
    """
    Detect and remove outliers from all numerical columns in the DataFrame using either the IQR or Z-score method.
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    method (str): The method to use for outlier detection, either 'iqr' or 'zscore'. Defaults to 'iqr'.
    Returns:
    pd.DataFrame: A DataFrame with outliers removed.
    """
    df_clean = df.copy()  # Copy the original DataFrame to avoid modifying it directly
    # df_clean.fillna(df_clean.mean(), inplace=True)
    if method == "iqr":
        # Loop through each numeric column in the DataFrame
        for column in df_clean.select_dtypes(include=[np.number]).columns:
            Q1 = df_clean[column].quantile(0.25)  # First quartile (25th percentile)
            Q3 = df_clean[column].quantile(0.75)  # Third quartile (75th percentile)
            IQR = Q3 - Q1  # Interquartile range
            # Calculate lower and upper bounds for detecting outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
             # Debugging output
            print(f"Column: {column}")
            print(f"  Q1: {Q1}, Q3: {Q3}, IQR: {IQR}, Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")           
            # Remove rows with outliers for the current column
            df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    elif method == "zscore":
        # Apply Z-score method for detecting outliers
        z_scores = np.abs(stats.zscore(df_clean.select_dtypes(include=[np.number])))
        df_clean = df_clean[(z_scores < 3).all(axis=1)]
    else:
        raise ValueError("Method must be either 'iqr' or 'zscore'")
    return df_clean
    