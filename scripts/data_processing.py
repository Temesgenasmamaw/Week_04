import pandas as pd
import numpy as np
import scipy.stats 
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