#!/usr/bin/env python
import glob  # retrieving all file paths matching a pattern
import pandas as pd  # data processing
import os  # executing operating system tasks
import numpy as np  # working with arrays
pd.options.mode.chained_assignment = None


def is_null(df):
    print("-" * 50)
    print("Missing values")
    print(df.isna().sum()) # Calculate the sum of missing values for each column
    print("Sum of missing values:", df.isna().sum().sum()) # Calculate the total sum of missing values in the DataFrame
    print("-" * 50)


def join_all_files():
    project_folder = os.getcwd()  # Get the current working directory (project folder)
    data_folder = 'data'  # Name of the folder where the files are located
    files = os.path.join(project_folder, data_folder, 'real_estate_data_*.csv')
    files = glob.glob(files)  # Get a list of all matching file paths
    all_df = pd.concat(map(pd.read_csv, files), ignore_index=True) # Read and concatenate all CSV files into a single DataFrame

    print(all_df.shape[0]) # Print the number of rows in the DataFrame before column renaming
    name_mapping = {
        'nazwa': 'name',
        'dzielnica': 'district',
        'cena': 'price',
        'metry': 'area',
        'pietro': 'floor',
        'pokoje': 'rooms',
        'rok_budowy': 'year_built',
        'link': 'link',
        'data_oferty': 'offer_date',
        'mieszkanie': 'apartment',
        'dom': 'house'
    }
    all_df = all_df.rename(columns=name_mapping) # Rename the columns using the name_mapping dictionary
    return all_df

# ----------------------------------------------------------------------------------------------------------------------
# Processing data for visualization and analysis
# ----------------------------------------------------------------------------------------------------------------------


def clean_data_common(df):
    df.drop_duplicates(inplace=True)
    df = df[df['apartment'] == 1]  # Keep only rows where 'apartment' column is 1 (indicating an apartment)
    # Removing user errors
    df = df[df['year_built'] >= 1860]  # Keep only rows where 'year_built' is greater than or equal to 1860 - date of the first building in the city
    df = df[df['price'] <= 23000000]  # Keep only rows where 'price' is less than or equal to 23000000 - highest price for apartament in Poland
    # Removing locations that are not in the city's territory
    df = df[~df['district'].str.contains('Piaśniki Świętochłowice')]  # Exclude rows containing 'Piaśniki Świętochłowice' in 'district' column
    df = df[~df['district'].str.contains('Siemianowice')]  # Exclude rows containing 'Siemianowice' in 'district' column
    df['district'] = df['district'].str.strip()  # Remove leading and trailing whitespaces from 'district' column
    df['district'] = df['district'].replace("Środmieście", "Śródmieście")  # Replace 'Środmieście' with 'Śródmieście' in 'district' column
    df['district'] = df['district'].replace("OS.WITOSA", "Witosa")  # Replace 'OS.WITOSA' with 'Witosa' in 'district' column
    df = df.drop(columns=['apartment', 'house'])  # Drop the 'apartment' and 'house' columns from the DataFrame
    df = df.replace(['-', ' ', ''], np.nan)  # Replace occurrences of '-', ' ', and empty strings with NaN values
    df['offer_date'] = pd.to_datetime(df['offer_date'])  # Convert 'offer_date' column to datetime format
    df.rooms = df['rooms'].astype(float) # Convert 'rooms' column to integer data type
    return df


def clean_data(df):
    df = clean_data_common(df)  # Apply common data cleaning steps to the DataFrame
    df.dropna(inplace=True)  # Drop rows with missing values from the DataFrame
    return df


def clean_data_drop(df):
    df = clean_data_common(df)
    df = df.drop(columns=['floor', 'year_built'])
    df.dropna(inplace=True) # Drop rows with missing values from the DataFrame
    return df


def clean_data_fillna(df):
    df = clean_data_common(df)
    df.rooms.fillna(df.rooms.mean(), inplace=True)
    df.area.fillna(df.area.mean(), inplace=True)
    df.price.fillna(df.price.mean(), inplace=True)
    return df


def outliers(df):
    df_numeric = df.select_dtypes(include='number')  # Wybieramy tylko kolumny numeryczne
    q1 = df_numeric.quantile(0.25)
    q3 = df_numeric.quantile(0.75)
    iqr = q3 - q1
    l_boundary = (q1 - 1.5 * iqr)
    u_boundary = (q3 + 1.5 * iqr)
    col_names = iqr.index
    num_l = (df_numeric[col_names] < l_boundary)
    print("-" * 50)
    print(f'Number of outliers below: \n{num_l.sum()}')
    print("-" * 50)
    num_u = (df_numeric[col_names] > u_boundary)
    print(f'Number of outliers above: \n{num_u.sum()}')
    print("-" * 50)
    df_outliers = pd.DataFrame({'first_quarter': l_boundary, 'third_quarter': u_boundary})
    print(df_outliers)
    print("-" * 50)
    print(f'Data with outlier observations: {df.shape[0]}')
    print("-" * 50)
    for col, row in df_outliers.iterrows():
        df = df[(df[col] >= row['first_quarter']) & (df[col] <= row['third_quarter'])]
    print(f'Data after removing outlier observations: {df.shape[0]}')
    print("-" * 50)
    return df


# ----------------------------------------------------------------------------------------------------------------------
# Adding averages, calculating the number of bids, creating teams of districts and their colors
# ----------------------------------------------------------------------------------------------------------------------


def add_means(df):
    df['avg_price_area'] = df.price / df.area
    df['avg_price_rooms'] = df.price / df.rooms
    return df


def offers_count(df):
    grouped_df = df.groupby('offer_date')
    count_df = grouped_df.count()
    print(count_df)


def set_cat(df):
    mid = ['Bogucice', 'Koszutka', 'Muchowiec', 'Paderewskiego', 'Śródmieście', 'Paderewskiego-Muchowiec']
    north = ['Dąb', 'Dębowe Tarasy', 'Józefowiec', 'Witosa', 'Tysiąclecia', 'Załęże']
    west = ['Brynów', 'Kokociniec', 'Ligota', 'Ligota-Panewniki', 'Panewniki', 'Ptasie', 'Wełnowiec-Józefowiec',
            'Wełnowiec', 'Zadole', 'Zgrzebnioka', 'Załęska Hałda-Brynów', 'Brynów-Zgrzebnioka', 'Nowa Ligota',
            'Brynów-Osiedle Zgrzebnioka']
    east = ['Burowiec', 'Dąbrówka Mała', 'Dąbrowa Mała', 'Giszowiec', 'Janów', 'Janów-Nikiszowiec', 'Nikiszowiec',
            'Szopienice-Burowiec', 'Szopienice', 'Walentego Roździeńskiego', 'Zawodzie']
    south = ['Kostuchna', 'Murcki', 'Młodych', 'Ochojec', 'Odrodzenia', 'Piotrowice-Ochojec', 'Podlesie', 'Zarzecze',
             'Piotrowice', 'Radockiego']
    cat = []
    for row in df['district']:
        if row in mid:
            cat.append('center')
        elif row in north:
            cat.append('north')
        elif row in west:
            cat.append('west')
        elif row in east:
            cat.append('east')
        elif row in south:
            cat.append('south')
        else:
            cat.append('unknow')
    df['category'] = cat
    return df


def get_color(df):
    df = set_cat(df)
    colours = {'center': "#7c1158", 'north': "#4421af", 'west': "#0d88e6", 'east': "#5ad45a", 'south': "#ebdc78", 'unknow': "#b30000"}
    df = df['category'].replace(colours)
    return df


def main():
    all_df = join_all_files()
    is_null(all_df)
    offers_count(all_df)
    print(all_df.describe().to_string())
    # ------------------------------------------------------------------------------------------------------------------
    # Cleaned and converted data
    # ------------------------------------------------------------------------------------------------------------------
    clean_df = clean_data(all_df)
    print(clean_df.describe().to_string())
    # ------------------------------------------------------------------------------------------------------------------
    # Data operations
    # ------------------------------------------------------------------------------------------------------------------
    print(clean_df.describe())
    print(clean_df.groupby(['district'], as_index=False).count())
    print("-" * 50)
    print("Missing values")
    print(clean_df.isna().sum())
    print("Sum of missing values:", clean_df.isna().sum().sum())
    print("-" * 50)


if __name__ == '__main__':
    main()
