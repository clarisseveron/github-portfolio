#!/usr/bin/env python
import glob  # retrieving all file paths matching a pattern
import datetime as dt  # handling time-related tasks
import pandas as pd  # data processing
import os  # executing operating system tasks
import numpy as np  # working with arrays
from matplotlib import pyplot as plt  # data visualization
import seaborn as sns  # data visualization
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso  # creating regression models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # model fit evaluation
from sklearn.model_selection import train_test_split, cross_val_score  # creating train and test sets, model evaluation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler  # data standardization

# way to disable the warning that Pandas generates when performing chained assignments.
# (i.e., multiple assignments in a single line of code)
from sklearn.tree import DecisionTreeRegressor

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
    is_null(all_df)
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


# ----------------------------------------------------------------------------------------------------------------------
# Data visualization
# ----------------------------------------------------------------------------------------------------------------------


def all_data_analysis(df):
    numeric_columns = df.select_dtypes(include=['int', 'float']).columns
    mean_df = df.groupby('link')[numeric_columns].mean().reset_index()
    sns.pairplot(mean_df, kind="reg")
    plt.ticklabel_format(style="plain", useOffset=False)
    plt.savefig('img/pairplot.png')
    plt.show()


def distribution(df):
    sns.histplot(df['price'], kde=True, color='red', edgecolor="blue", linewidth=2)
    plt.title(f"Distribution")
    plt.savefig(f'img/distribution.png')
    plt.show()


def heatmap(df, name):
    df = df.select_dtypes(include='number')
    sns.heatmap(df.corr(), annot=True, cmap='magma')
    plt.title(f"heatmap of data correlation {name}")
    plt.savefig(f'img/heatmap_{name}.png')
    return plt


def offers_by_rooms(df):
    mean_df = df.groupby(['rooms']).count().reset_index()
    x = mean_df.rooms
    y = mean_df.link
    plt.bar(x, y, color='green')
    plt.title("number of offers in relation to the number of rooms")
    plt.xlabel('rooms')
    plt.ylabel('number of offers')
    return plt


def number_of_offers_by_date(df):
    grouped_df = df.groupby('offer_date')
    mean_df = grouped_df.mean(numeric_only=True).reset_index()
    count_df = grouped_df.count()
    x = mean_df.offer_date
    y = count_df.link
    plt.plot(x, y, label='number of offers in relation to the date', color='sandybrown')
    plt.title("number of offers on a particular day")
    plt.xlabel('offer date')
    plt.ylabel('number of offers')
    y_mean = [np.mean(y)] * len(x)
    plt.plot(x, y_mean, label='average number of offers by date', linestyle='--', color='sandybrown')
    plt.legend(loc=4)
    plt.xticks(rotation=90)
    return plt


def mean_price_by_rooms(df):
    mean_df = df.groupby(['rooms', 'link'], as_index=False).mean(numeric_only=True)
    mean_df = mean_df.groupby('rooms')['avg_price_area'].mean().reset_index()
    x = mean_df.rooms
    y = mean_df.avg_price_area
    plt.plot(x, y, label='average price per sqm for the number of rooms', color='green')
    plt.plot(x, y, 'o', color='green')
    plt.title("Graph of the relationship of price and price per sqm in relation to the number of rooms")
    plt.xlabel('rooms')
    plt.ylabel('average price per sqm')
    y_mean = [np.mean(y)] * len(x)
    plt.plot(x, y_mean, label='average price per sqm for all offers', linestyle='--', color='green')
    plt.legend(loc=0)
    plt2 = plt.twinx()
    mean_df = df.groupby(['rooms', 'link'], as_index=False).mean(numeric_only=True).groupby('rooms')['price'].mean().reset_index()
    x = mean_df.rooms
    y = mean_df.price
    plt2.plot(x, y, label='average price for number of rooms', color='purple')
    plt2.plot(x, y, 'o', color='purple')
    plt2.set_ylabel('average price')
    y_mean = [np.mean(y)] * len(x)
    plt.plot(x, y_mean, label='average price for all offers', linestyle='--', color='purple')
    plt.legend(loc=4)
    return plt


def mean_price_by_date(df):
    numeric_columns = df.select_dtypes(include=['int', 'float']).columns
    mean_df = df.groupby('offer_date')[numeric_columns].mean().reset_index()
    x = mean_df.offer_date
    y = mean_df.avg_price_area
    plt.plot(x, y, label='average price per sqm for the date', color='orange')
    plt.title("average number of sqm and their price by date")
    plt.xlabel('offer date')
    plt.ylabel('price per sqm')
    y_mean = [np.mean(y)] * len(x)
    plt.plot(x, y_mean, label='average price per sqm for all dates', linestyle='--', color='orange')
    plt.legend(loc=2)
    plt.xticks(rotation=90)
    plt2 = plt.twinx()
    x = mean_df.offer_date
    y = mean_df.area
    plt2.plot(x, y, label='average number of sqm for the date', color='green')
    plt2.set_ylabel('sqm')
    y_mean = [np.mean(y)] * len(x)
    plt.plot(x, y_mean, label='average number of sqm of all dates', linestyle='--', color='green')
    plt.legend(loc=4)
    plt.savefig(f'img/price_by_date.png', bbox_inches='tight')
    return plt


def mean_offers_by_rooms_in_one(df, rooms):
    mean_df = df.groupby('offer_date').count().reset_index()
    x = mean_df.offer_date
    y = mean_df.link
    plt.plot(x, y, label=f'number of offers for {rooms} room/rooms for the date')
    plt.title("number of offers split by number of rooms in relation to date")
    plt.xlabel('offer date')
    plt.ylabel('number of offers')
    plt.legend(loc=7)
    plt.xticks(rotation=90)
    return plt


def mean_area_by_rooms_in_one(df, rooms):
    mean_df = df.groupby('offer_date').mean(numeric_only=True).reset_index()
    x = mean_df.offer_date
    y = mean_df.area / rooms
    plt.plot(x, y, label=f'Average sqm per room for {rooms} rooms for the date')
    plt.title("average sqm per room split by number of rooms in relation to date")
    plt.xlabel('offer date')
    plt.ylabel('sqm per room')
    plt.legend(loc=7)
    plt.xticks(rotation=90)
    return plt


def mean_price_by_rooms_in_one(df, rooms):
    mean_df = df.groupby('offer_date').mean(numeric_only=True).reset_index()
    x = mean_df.offer_date
    y = mean_df.avg_price_area
    plt.plot(x, y, label=f'average price per {rooms} rooms for the date')
    plt.title("average price per sqm split by number of rooms in relation to date")
    plt.xlabel('offer date')
    plt.ylabel('price per sqm')
    plt.legend(loc=2)
    plt.xticks(rotation=90)
    return plt


def mean_area_by_district(df, room):
    mean_df = df.groupby(['district', 'link'], as_index=False).mean(numeric_only=True)
    mean_df = mean_df.groupby('district')[['avg_price_area', 'area']].mean()
    mean_df = mean_df.reset_index()
    mean_df = mean_df.sort_values(by=['avg_price_area'], ascending=True).reset_index()
    mean_df = mean_df[['district', 'avg_price_area', 'area']]
    plt.bar(mean_df.district, mean_df.avg_price_area, width=0.4, label='average price per sqm for the district',
            color=get_color(mean_df))
    plt.title(f"average price for a {room}-room apartment and sqm in a given district")
    plt.xlabel('district')
    plt.ylabel('price per sqm')
    y_mean = [np.mean(mean_df.avg_price_area)] * len(mean_df.district)
    plt.plot(mean_df.district, y_mean, label='average for all districts', linestyle='--', color='blue')
    plt.legend(loc=0)
    plt.xticks(rotation=90)
    plt2 = plt.twinx()
    x = mean_df.index
    y = mean_df.area
    plt.bar(x, y, alpha=0.5, label='average number of sqm for the district', color='grey')
    plt2.set_ylabel('number of sqm')
    y_mean = [np.mean(y)] * len(x)
    plt.plot(x, y_mean, label='average number of sqm for all offers', linestyle='--', color='black')
    plt.legend(loc=4)
    return plt


def district_histogram(df, room):
    mean_df = df.groupby(['district', 'link'], as_index=False).mean(numeric_only=True).groupby(['district']).count().reset_index()
    mean_df = mean_df.sort_values(by=['link'], ascending=True).reset_index()
    x = mean_df.district
    y = mean_df.link
    plt.title(f"number of offers of {room}-room apartment in a given district")
    plt.bar(x, y, label='number of offers in a given district', color=get_color(mean_df))
    plt.xlabel('district')
    plt.xticks(rotation=90)
    return plt


def area_by_district(df, room):
    mean_df = df.groupby(['district', 'link'], as_index=False).mean(numeric_only=True)
    mean_df = mean_df.groupby('district')['year_built'].mean().reset_index()
    mean_df = mean_df.sort_values(by=['year_built'], ascending=True).reset_index()
    x = mean_df.index
    y = mean_df.year_built
    colors = get_color(mean_df)
    plt.scatter(mean_df.district, mean_df.year_built, label='average year of build for the district', color=colors)
    plt.title(f"average year of build for {room}-room apartment in the district")
    plt.xlabel('district')
    plt.ylabel('year of build')
    y_mean = [np.mean(y)] * len(x)
    plt.plot(x, y_mean, label='average for all districts', linestyle='--', color='green')
    plt.legend(loc=0)
    plt.xticks(rotation=90)
    plt.grid(True)
    return plt


def data_visualization(df, plt=plt):
    plt.rcParams["figure.figsize"] = (20, 8)
    plt = mean_price_by_date(df)
    plt.subplots_adjust(bottom=0.2, wspace=0.5)
    plt.margins(0.2)
    plt.show()

    plt.subplot(1, 3, 1)
    plt = offers_by_rooms(df)
    plt.subplot(1, 3, 2)
    plt = number_of_offers_by_date(df)
    plt.subplot(1, 3, 3)
    plt = mean_price_by_rooms(df)
    plt.subplots_adjust(bottom=0.2, wspace=0.3)
    plt.margins(0.2)
    plt.tight_layout()
    plt.savefig('img/general_visualization.png', bbox_inches='tight')
    plt.show()

    for x in range(1, 5):
        df_by_room = df[df.rooms == x]
        plt = mean_offers_by_rooms_in_one(df_by_room, x)
        plt.grid(True)
        plt.subplots_adjust(bottom=0.2)
        plt.savefig('img/rooms_chart_1.png', bbox_inches='tight')
    plt.show()

    for x in range(1, 5):
        df_by_room = df[df.rooms == x]
        plt = mean_area_by_rooms_in_one(df_by_room, x)
        plt.grid(True)
        plt.subplots_adjust(bottom=0.2)
        plt.savefig('img/rooms_chart_2.png', bbox_inches='tight')
    plt.show()

    for x in range(1, 5):
        df_by_room = df[df.rooms == x]
        plt = mean_price_by_rooms_in_one(df_by_room, x)
        plt.grid(True)
        plt.subplots_adjust(bottom=0.2)
        plt.savefig('img/rooms_chart_3.png', bbox_inches='tight')
    plt.show()


def data_visualization_by_rooms(df, plt=plt):
    df = df[df.district != 'śląskie']
    for x in range(1, 5):
        df_by_room = df[df.rooms == x]
        plt.rcParams["figure.figsize"] = (20, 8)
        plt.margins(0.6)
        plt.subplot(1, 3, 1)
        plt = mean_area_by_district(df_by_room, x)
        plt.grid(axis='y')
        plt.subplot(1, 3, 2)
        plt = district_histogram(df_by_room, x)
        plt.subplot(1, 3, 3)
        plt = area_by_district(df_by_room, x)
        plt.subplots_adjust(bottom=0.3, wspace=0.3)
        plt.savefig(f'img/{x}_room(s).png', bbox_inches='tight')
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# Predictive capability analysis
# ----------------------------------------------------------------------------------------------------------------------


def evaluation(y, predictions):
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    r_mse = np.sqrt(mean_squared_error(y, predictions))
    r_sqr = r2_score(y, predictions)
    return mae, mse, r_mse, r_sqr


def models_verification(df): # old function
    num = ['area', 'rooms', 'offer_date', 'year_built', 'floor']
    cat = ['district']
    df.offer_date = df.offer_date.map(dt.datetime.toordinal)
    X = num + cat
    y = df['price']
    X = df[X]
    X = pd.get_dummies(X, 'district')  # Encoding cat features in dataset X using the One-Hot Encoding method
    scaler = StandardScaler()
    X[num] = scaler.fit_transform(X[num])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = pd.DataFrame(columns=["Model", "MAE", "MSE", "RMSE", "R2 Score", 'RMSE Cross-Validation'])
    m = [LinearRegression, Ridge, ElasticNet, Lasso]
    print("Selected models:" + str(m))
    models = ['LinearRegression()',
              'Ridge()',
              'Ridge(alpha=0.001)',
              'Ridge(alpha=0.01)',
              'Ridge(alpha=0.1)',
              'Ridge(alpha=1)',
              'Ridge(alpha=5)',
              'Ridge(alpha=10)',
              'Lasso(tol=1.5+15)',
              'Lasso(alpha=0.001, tol=1.5+15)',
              'Lasso(alpha=0.01, tol=1.5+15)',
              'Lasso(alpha=0.1, tol=1.5+15)',
              'Lasso(alpha=1, tol=1.5+15)',
              'Lasso(alpha=5, tol=1.5+15)',
              'Lasso(alpha=10, tol=1.5+15)',
              'Lasso(alpha=15, tol=1.5+15)',
              'Lasso(alpha=20, tol=1.5+15)',
              'Lasso(alpha=100, tol=1.5+15)',
              'ElasticNet(alpha=0.001)',
              'ElasticNet(alpha=0.01)',
              'ElasticNet(alpha=0.1)',
              'ElasticNet(alpha=1)',
              ]
    for x in models:
        model = eval(x)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae, mse, r_mse, r_sqr = evaluation(y_test, predictions)
        rmse_cross_val = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=10)).mean()
        new_row = pd.DataFrame({"Model": [x], "MAE": [mae], "MSE": [mse], "RMSE": [r_mse], "R2 Score": [r_sqr],
                                'RMSE Cross-Validation': [rmse_cross_val]})
        results = pd.concat([results, new_row], ignore_index=True)
    print(results.sort_values(by=['RMSE Cross-Validation']).to_string())


def knn(df):
    num = ['area', 'rooms', 'offer_date', 'year_built', 'floor']
    cat = ['district']
    df.offer_date = df.offer_date.map(dt.datetime.toordinal)
    X = num + cat
    y = df['price']
    X = df[X]
    X = pd.get_dummies(X, 'district')  # Encoding cat features in dataset X using the One-Hot Encoding method
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(X_train,y_train)
    prediction = knn.predict(X_test)
    #print('Prediction: {}'.format(prediction))
    print('With KNN (K=1) accuracy is: ',knn.score(X_test,y_test)) # accuracy
    # Model complexity
    neig = np.arange(1, 25)
    train_accuracy = []
    test_accuracy = []
    # Loop over different values of k
    for i, k in enumerate(neig):
        # k from 1 to 25(exclude)
        knn = KNeighborsClassifier(n_neighbors=k)
        # Fit with knn
        knn.fit(X_train,y_train)
        #train accuracy
        train_accuracy.append(knn.score(X_train, y_train))
        # test accuracy
        test_accuracy.append(knn.score(X_test, y_test))

    # Plot
    plt.figure(figsize=[13,8])
    plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
    plt.plot(neig, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.title('-value VS Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.xticks(neig)
    plt.savefig('graph.png')
    plt.show()
    print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))


def only_linear(df):
    df = set_cat(df)
    num = ['area', 'rooms', 'offer_date', 'year_built', 'floor']
    cat = ['category'] # No need for coeff
    df.offer_date = df.offer_date.map(dt.datetime.toordinal)
    X = num + cat
    y = df['price']
    X = df[X]
    X = pd.get_dummies(X, 'district')  # Encoding cat features in dataset X using the One-Hot Encoding method
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    # Displaying the Intercept
    print(lm.intercept_)
    coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
    print(coeff_df)
    predictions = lm.predict(X_test)
    plt.scatter(y_test, predictions, edgecolor='black')
    plt.show()
    sns.histplot((y_test - predictions), bins=50, kde=True, edgecolor="black", linewidth=1, color='blue')
    plt.show()
    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    print('RMSE Cross-Validation:', np.sqrt(-cross_val_score(lm, X, y, scoring="neg_mean_squared_error", cv=10)).mean())


def max_leaf_nodes(df):
    df = set_cat(df)
    num = ['area', 'rooms', 'offer_date', 'year_built', 'floor']
    cat = ['category'] # No need for coeff
    df.offer_date = df.offer_date.map(dt.datetime.toordinal)
    X = num + cat
    y = df['price']
    X = df[X]
    X = pd.get_dummies(X, 'district')  # Encoding cat features in dataset X using the One-Hot Encoding method
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for max_leaf_nodes in [5, 50, 500, 5000, 15000]:
        mae = get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, mae))


# ----------------------------------------------------------------------------------------------------------------------


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


def main():
    all_df = join_all_files()
    offers_count(all_df)
    print(all_df.describe().to_string())

    # ------------------------------------------------------------------------------------------------------------------
    # Cleaned and converted data
    # ------------------------------------------------------------------------------------------------------------------
    clean_df = clean_data(all_df)
    print(clean_df.describe().to_string())
    '''
    all_data_analysis(clean_df)
    heatmap(clean_df, 'clean_df').show()
    distribution(clean_df)

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

    # ------------------------------------------------------------------------------------------------------------------
    # Correlation with dropped missing values and non-correlated data
    # ------------------------------------------------------------------------------------------------------------------
    clean_df_drop_na = clean_data_drop(all_df)
    heatmap(clean_df_drop_na, 'clean_df_drop').show()

    # ------------------------------------------------------------------------------------------------------------------
    # Correlation with filled missing values and non-correlated data
    # ------------------------------------------------------------------------------------------------------------------
    clean_df_fill_na = clean_data_fillna(all_df)
    heatmap(clean_df_fill_na, 'clean_df_fill_na').show()
    '''
    # ------------------------------------------------------------------------------------------------------------------
    # Correlation after removing outliers
    # ------------------------------------------------------------------------------------------------------------------
    out_of_outliers = outliers(clean_df)
    #distribution(out_of_outliers)
    #heatmap(out_of_outliers, 'out_of_outliers').show()

    # ------------------------------------------------------------------------------------------------------------------
    # Visualization and prediction
    # ------------------------------------------------------------------------------------------------------------------
    clean_df = add_means(clean_df)
    #data_visualization(clean_df)
    #data_visualization_by_rooms(clean_df)
    #models_verification(clean_df) # old function
    #models_verification(out_of_outliers) # old function
    knn(clean_df)
    knn(out_of_outliers)
    only_linear(clean_df)
    only_linear(out_of_outliers)


if __name__ == '__main__':
    main()
