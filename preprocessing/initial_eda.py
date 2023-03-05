"""
Module to perform initial eda on merged dataset from data_join.py to get initial idea about the dataset for preprocessing and further computations.
imports: numpy, pandas
"""
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_data():
    """
    function to read csv file
    parameters: None
    return: data frames fraud, beneficiary, inpatient, outpatient.
    raise FileExistsError: raises an exception when file is not found
    """
    try:
        merged = pd.read_csv("data/preprocessed.csv")
    except FileExistsError as error:
        raise error
    return merged


def get_unique_values(dataframe):
    """
    function to get unique values of dataframe
    parameters: merged dataset
    return: None
    """
    unique_vals = dataframe.nunique().sort_values()
    print("Number of unique values:\n", unique_vals)


def get_dimension(dataframe):
    """
    function to get number of rows and columns, all columns, to show first 5 data.
    parameters: merged dataset
    return: None    
    """
    print(dataframe.describe())
    print("Number of rows and columns:", dataframe.shape)
    print("\nColumn names:\n", dataframe.columns)
    print("\nFirst  rows:\n", dataframe.head())


def get_missing_data(dataframe):
    """
    function to prints the percentage of missing values in each column.
    parameters: merged dataset
    return: None
    """
    total_missing = dataframe.isnull().sum().sort_values(ascending=False)
    percent_missing = (total_missing / len(dataframe)) * 100
    missing_data = pd.concat([total_missing, percent_missing], axis=1, keys=['Total', 'Percent'])
    print("Percentage of missing values:\n", missing_data)


def plot_heatmap(dataframe):
    """
    function to plot heat map 
    parameters: merged dataset
    return: None
    """
    corr = dataframe.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()


def plot_boxplots(dataframe, columns):
    """
    function to plot boxplots
    parameters: merged dataset, columns for which we want to plot 
    return: None
    """
    for column in columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='PotentialFraud', y=column, data=dataframe)
        plt.title(f"{column} by PotentialFraud")
        plt.show()


def get_admit_analysis(dataframe):
    """
    function to get range of admitdays
    parameters: merged dataset, columns for which we want to plot 
    return: None
    """
    ## Lets check Min and Max values of AdmitforDays column in Train and Test.
    print('Min Admit:- ', dataframe.AdmitForDays.min())
    print('Max Admit:- ', dataframe.AdmitForDays.max())
    dataframe.AdmitForDays.isnull().sum()  # Check Null values.


def get_correlation(dataframe):
    """
    function get correlation
    parameters: merged dataset
    return: top 5 columns having highest correlation with PotentialFraud
    """
    corr_to_target = dataframe.corr()['PotentialFraud'].abs().sort_values(ascending=False)
    return corr_to_target[1:6].index


def get_eda():
    """
    function get complete eda
    parameters: None
    return: None
    """
    merged = read_data()
    saved = merged
    get_unique_values(merged)
    get_dimension(merged)
    get_missing_data(merged)
    get_admit_analysis(merged)
    plot_heatmap(merged)
    top_5_corr = get_correlation(merged)
    plot_boxplots(saved, top_5_corr)


def main():
    """
    main function
    parameters: None
    return: None
    """
    get_eda()


if __name__ == "__main__":
    """
    main function
    parameters: None
    return: None
    """
    main()
