#!/usr/bin/env python
import pandas as pd  # data processing
import numpy as np  # working with arrays
from matplotlib import pyplot as plt  # data visualization
import seaborn as sns  # data visualization
import preprocessing
pd.options.mode.chained_assignment = None


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

def linear(df, name='Linear'):
    df = preprocessing.set_cat(df)
    num = ['area', 'rooms', 'offer_date', 'year_built', 'floor']
    cat = ['category']
    X = num + cat
    y = df['price']
    X = df[X]
    X = pd.get_dummies(X, 'district')  # Encoding cat features in dataset X using the One-Hot Encoding method
    scaler = StandardScaler()
    X[num] = scaler.fit_transform(X[num])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    # Displaying the Intercept
    print(lm.intercept_)
    coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
    print(coeff_df)
    predictions = lm.predict(X_test)
    print('R^2:', r2_score(y_test, predictions))
    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    plt.scatter(y_test, predictions, edgecolor='black')
    plt.xlabel("Prices")
    plt.ylabel("Predicted prices")
    plt.title(f"{name} - Prices vs Predicted prices")
    plt.savefig(f'img/price_predicted_{name}.png', bbox_inches='tight')
    plt.show()
    sns.histplot((y_test - predictions), bins=50, kde=True, edgecolor="black", linewidth=1, color='blue')
    plt.title(f"{name} - Distribution of model prediction errors")
    plt.savefig(f'img/distribution_of_errors_{name}.png', bbox_inches='tight')
    plt.show()
    return np.sqrt(metrics.mean_squared_error(y_test, predictions)), r2_score(y_test, predictions)

def main():
    all_df = preprocessing.join_all_files()
    preprocessing.offers_count(all_df)
    # ------------------------------------------------------------------------------------------------------------------
    # Cleaned and converted data
    # ------------------------------------------------------------------------------------------------------------------
    clean_df = preprocessing.clean_data(all_df)
    all_data_analysis(clean_df)
    heatmap(clean_df, 'clean_df').show()
    distribution(clean_df)
    # ------------------------------------------------------------------------------------------------------------------
    # Correlation with dropped missing values and non-correlated data
    # ------------------------------------------------------------------------------------------------------------------
    clean_df_drop_na = preprocessing.clean_data_drop(all_df)
    heatmap(clean_df_drop_na, 'clean_df_drop').show()
    # ------------------------------------------------------------------------------------------------------------------
    # Correlation with filled missing values and non-correlated data
    # ------------------------------------------------------------------------------------------------------------------
    clean_df_fill_na = preprocessing.clean_data_fillna(all_df)
    heatmap(clean_df_fill_na, 'clean_df_fill_na').show()
    # ------------------------------------------------------------------------------------------------------------------
    # Correlation after removing outliers
    # ------------------------------------------------------------------------------------------------------------------
    out_of_outliers = preprocessing.outliers(clean_df)
    distribution(out_of_outliers)
    heatmap(out_of_outliers, 'out_of_outliers').show()


if __name__ == '__main__':
    main()
