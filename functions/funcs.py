import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.style as style
import seaborn as sns


from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_samples, silhouette_score

os.chdir("C:/Users/migue/OneDrive - NOVAIMS/Data Science/Coding Courses/Machine Learning II/Project")
wd = os.getcwd()

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


def create_dfs(path: str) -> dict:
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
            print(f"Created dataframe {blue}{name}{end} from {file}")

        elif file.endswith(".xlsx"):
            globals()[name] = pd.read_excel(path + file)
            print(f"Created dataframe {blue}{name}{end} from {file}")

        else:
            print(f"File {file} is not a .csv or .xlsx file. Skipping it.")

    # Return a dictionary with the name of the dataframe as the key and the dataframe as the value
    return {k: v for k, v in globals().items() if isinstance(v, pd.DataFrame)}

    
def drop_univariate_cols(df: pd.DataFrame, df_name: str, drop_all: bool = False) -> pd.DataFrame:
    """Drops the columns that have only one unique value in a dataframe.
        Asks user for confirmation before dropping the column if drop_all is False.
    Args:
        df (pd.DataFrame): The dataframe to be analyzed.
        df_name (str): Name of the dataframe.
    Returns:
        pd.DataFrame: The dataframe with the columns that have only one unique value dropped.
    """

    # Loop through each column and check if there is only one unique value in it.
    for i in df.columns:
        if len(df[i].unique()) == 1:
            if drop_all:
                df = df.drop(i, axis=1)
                print(f"Dropped column {blue}{i}{end} from {blue}{df_name}{end} because it had only one unique value.")
            else:
                drop = input(f"Column {blue}{i}{end} has only one unique value. Do you want to drop it? (y/n): ")
                if drop.lower() == "y":
                    df = df.drop(i, axis=1)
                    print(f"Dropped column {blue}{i}{end} from {blue}{df_name}{end} because it had only one unique value.")
                elif drop.lower() == "n":
                    print(f"Did not drop column {blue}{i}{end} from {blue}{df_name}{end}.")
                else:
                    print("Please enter y or n.")

    return df
    

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


def save_to_csv(df_dict: dict, path: str = wd + "/data/", index: bool = False) -> None:
    """Saves the dataframes inside df_dict to separate .csv file in the path folder with the key as the file name. 
    Args:
        df_dict (dict): A dictionary with the name of the file to be created as the key and the dataframe as the value.
        path (str): The path to the folder where the files will be saved.
        index (bool, optional): Whether to save the index of the dataframe. Defaults to False.
    Returns:
        None
    """
    for key in df_dict:
        pd.DataFrame(df_dict[key]).to_csv(path + key + '.csv', index=index)


def save_clusters(clusters: dict, path: str = wd + "/cluster_data/") -> None:
    """Saves the clusters to a csv file.

    Args:
        clusters (dict): The clusters to be saved.
    """
    
    
    for key in clusters:
        pd.DataFrame(clusters[key]).to_csv(path + key + '.csv', index=False)

    print(f"Clusters saved to {path}")



def create_clusters(data: pd.DataFrame,
                    clustering_functions: list,
                    clustering_params: list,
                    cluster_names: list = None,
                    scaler_functions: list = None,
                    scaler_names: list = None
                    ) -> dict:
    
    """ Creates clusters using the clustering functions and parameters provided.

    Args:
        data (pd.DataFrame): the data to be clustered.
        clustering_functions (list): the clustering functions to be used.
        clustering_params (list): list of parameters for each clustering functions.
        cluster_names (list, optional): The names of the clustering methods. Defaults to None.
        scaler_functions (list, optional): the scaling functions to transform the data. Defaults to None.
        scaler_names (list, optional): the names of these scaling functions. Defaults to None.

    Returns:
        _type_: A dictionary with the name of the clustering method as the key and the clustering object as the value.
    """
    
    clusters = {}
    for scaler_name, scaler in zip(scaler_names, scaler_functions):
        curr_data = scaler().fit_transform(data)

        for i, (cluster_name, cluster_func) in enumerate(zip(cluster_names, clustering_functions)):
            clusters[f'{cluster_name}_{scaler_name}'] = cluster_func(**clustering_params[i]).fit(curr_data)
        
        
    return clusters


# corr heatmap
def corr_heatmap(df: pd.DataFrame, title: str) -> None:
    """Plots a correlation heatmap for the dataframe provided.

    Args:
        df (pd.DataFrame): The dataframe to be plotted.
        title (str): The title of the plot.
    """
    corr = df.corr()
    plt.figure(figsize=(10,10))
    plt.title(title)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, mask=mask)
    plt.show()


# clustermap
def clustermap(df: pd.DataFrame, title: str) -> None:
    """Plots a clustermap for the dataframe provided.

    Args:
        df (pd.DataFrame): The dataframe to be plotted.
        title (str): The title of the plot.
    """
    corr = df.corr()
    plt.figure(figsize=(10,10))
    plt.title(title)
    sns.clustermap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.show()


def plot_silhouette(X: pd.DataFrame, range_n_clusters: list) -> None:
    """Plots the silhouette score for each cluster.
    
    Args:
        X (pd.DataFrame): The data to be clustered.
        range_n_clusters (list): The range of clusters to be used.
    Returns:
        None
        """
    silhouette_avg_n_clusters = []

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        silhouette_avg_n_clusters.append(silhouette_avg)
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

    plt.show()


    style.use("fivethirtyeight")
    plt.plot(range_n_clusters, silhouette_avg_n_clusters)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("silhouette score")
    plt.show()

