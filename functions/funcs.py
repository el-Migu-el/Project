import pandas as pd
import os

# Colors for printing
bold = "\033[1m"
blue = '\033[94m'
red = "\033[91m"
end = "\033[0m"

def print_na_cols(df: pd.DataFrame, df_name: str) -> None:    
    """Prints the number of missing values in each column of a dataframe.
    Args:
        df (pandas.DataFrame): The dataframe to be analyzed.
    Returns:
        None
    """

    na_cols = []
    # Loop through each column and check if there are any missing values in it.
    for i in df.columns:
        # If there are missing values, append the column name, the number of missing values and the percentage to the na_cols list.
        if df[i].isnull().sum() > 0:
            na_cols.append([i, df[i].isnull().sum(), round(df[i].isnull().sum() / len(df) * 100, 2)])

    if len(na_cols) > 0:
        print(f"In {blue}{bold}{df_name}{end}, the following columns have missing values:", end='\n\t')
        for i in na_cols:       
            print(f'-Column {blue}{i[0]}{end} has {red}{i[1]}{end} missing values. This equals {red}{i[2]}%{end} of its values.', end='\n\t')

    else:
        print(f"{blue}{bold}{df_name}{end} has no missing values.")
    
    print()

    
def print_cols(df: pd.DataFrame, df_name: str) -> None:
    """Prints the name of the columns in a dataframe.
    Args:
        df (pandas.DataFrame): The dataframe to be analyzed.
    Returns:
        None
    """

    # prints the name of the file coloring it in blue and bold
    print(f"{bold}Columns in {blue}{df_name}{end}{bold} are: {end}", end = "\n\t-")

        # iterates through the columns of the file
    for i, col in enumerate(df.columns):

        # checks if the column is the last one
        if i != len(df.columns) - 1:

            # changes line every 5 columns
            if i % 5 == 0 and i != 0:
                print(end = "\n\t-")

            # adds a comma after each column name is printed
            print(col, end = ", ")

        # if the column is the last one, adds a new line to switch to the next file
        else:
            print(col, end="\n\n")


def load_dfs(path: str) -> dict:
    """Creates a dict of dataframes from all the .csv and .xlsx files in a folder.
    Args:
        path (str): The path to the folder containing the files.
    Returns:
        dict: A dictionary with the name of the dataframe as the key and the dataframe as the value. 
    """

    # Read all files in the folder and create a dataframe for each one that ends with .csv or .xlsx
    for file in os.listdir(path):

        # if the file only has one word, use the name of the file as the name of the dataframe (without the extension)
        if len(file.split()) == 1:
            name = file.split(".")[0]
        else:
            name = file.split()[1]

        if file.endswith(".csv"):
            globals()[name] = pd.read_csv(path + file)
            print(f"Created dataframe {blue}{name}{end} for {file}")

        elif file.endswith(".xlsx"):
            globals()[name] = pd.read_excel(path + file)
            print(f"Created dataframe {blue}{name}{end} for {file}")

        else:
            print(f"File {file} is not a .csv or .xlsx file. Skipping it.")

    # Return a dictionary with the name of the dataframe as the key and the dataframe as the value
    return {k: v for k, v in globals().items() if isinstance(v, pd.DataFrame)}
     

def print_inf_cols(df: pd.DataFrame, df_name: str) -> None:
    """Prints the columns that have infinite values in a dataframe and how many they are.

    Args:
        df (pd.DataFrame): The dataframe to be analyzed.

        df_name (str): Name of the dataframe.

    Returns:
        None
    """
    inf_cols = []
    # Loop through each column and check if there are any infinite values in it.
    for i in df.columns:
        # If there are infinite values, append the column name, the number of infinite values and the percentage to the inf_cols list.
        if df[i].isin([float('inf'), float('-inf')]).sum() > 0:
            inf_cols.append([i, df[i].isin([float('inf'), float('-inf')]).sum(), round(df[i].isin([float('inf'), float('-inf')]).sum() / len(df) * 100, 2)])

    if len(inf_cols) > 0:
        print(f"In {blue}{bold}{df_name}{end}, the following columns have infinite values:", end='\n\t')
        for i in inf_cols:
            print(f'-Column {blue}{i[0]}{end} has {red}{i[1]}{end} infinite values. This equals {red}{i[2]}%{end} of its values.', end='\n\t')

    else:
        print(f"{blue}{bold}{df_name}{end} has no infinite values.")

    print()


def print_dup_cols(df: pd.DataFrame, df_name: str) -> None:
    """Prints the columns that have duplicate values in a dataframe and how many they are.

    Args:
        df (pd.DataFrame): The dataframe to be analyzed.

        df_name (str): Name of the dataframe.

    Returns:
        None
    """
    dup_cols = []
    # Loop through each column and check if there are any duplicate values in it.
    for i in df.columns:
        # If there are duplicate values, append the column name, the number of duplicate values and the percentage to the dup_cols list.
        if df[i].duplicated().sum() > 0:
            dup_cols.append([i, df[i].duplicated().sum(), round(df[i].duplicated().sum() / len(df) * 100, 2)])

    if len(dup_cols) > 0:
        print(f"In {blue}{bold}{df_name}{end}, the following columns have duplicate values:", end='\n\t')
        for i in dup_cols:
            print(f'-Column {blue}{i[0]}{end} has {red}{i[1]}{end} duplicate values. This equals {red}{i[2]}%{end} of its values.', end='\n\t')

    else:
        print(f"{blue}{bold}{df_name}{end} has no duplicate values.")

    print()


def create_educ_level(df: pd.DataFrame) -> list:
    """Takes the dataframe and the name of the dataframe and creates a new column with the education level of the respondent.
    This is done by looking at the prefix of the customer name column and assigning the education level accordingly.
    Bsc, Msc and PhD prefixes will be assigned to the education level column as 1, 2 and 3 respectively.
    If the prefix is not one of the three, the education level will be assigned as 0.

    Args:
        df (pd.DataFrame): The dataframe to be analyzed.

        df_name (str): Name of the dataframe.

    Returns:
        list: A list with the education level of each customer.
    """
    # Create a list with the prefixes of the customer names
    prefixes = [i.split()[0][:3].lower() for i in df['customer_name']]

    # Create a list with the education levels
    educ_level = []

    # Loop through the prefixes and assign the education level accordingly
    for i in prefixes:
        if i == 'bsc':
            educ_level.append(1)
        elif i == 'msc':
            educ_level.append(2)
        elif i == 'phd':
            educ_level.append(3)
        else:
            educ_level.append(0)

    # Return education level
    return educ_level


def save_to_csv(df_dict: dict, path: str) -> None:
    """Saves the dataframes inside df_dict to separate .csv file in the path folder with the key as the file name. 
    Args:
        df_dict (dict): A dictionary with the name of the file to be created as the key and the dataframe as the value.
        path (str): The path to the folder where the files will be saved.
    Returns:
        None
    """
    for key in df_dict:
        pd.DataFrame(df_dict[key]).to_csv(path + key + '.csv')

