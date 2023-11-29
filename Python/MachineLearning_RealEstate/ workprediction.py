#!/usr/bin/env python
import datetime as dt  # handling time-related tasks
import pandas as pd  # data processing
import numpy as np  # working with arrays
from matplotlib import pyplot as plt  # data visualization
import seaborn as sns  # data visualization
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso  # creating regression models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # model fit evaluation
from sklearn.model_selection import train_test_split, cross_val_score  # creating train and test sets, model evaluation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import svm
# way to disable the warning that Pandas generates when performing chained assignments.
# (i.e., multiple assignments in a single line of code)
pd.options.mode.chained_assignment = None

import preprocessing


# ----------------------------------------------------------------------------------------------------------------------
# Predictive capability analysis
# ----------------------------------------------------------------------------------------------------------------------


def evaluation(y, predictions):
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    r_mse = np.sqrt(mean_squared_error(y, predictions))
    r_sqr = r2_score(y, predictions)
    return mae, mse, r_mse, r_sqr


def models_verification(df):  # old function
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
    #df.offer_date = df.offer_date.map(dt.datetime.toordinal)
    X = num + cat
    y = df['price']
    X = df[X]
    X = pd.get_dummies(X, 'district')  # Encoding cat features in dataset X using the One-Hot Encoding method
    scaler = StandardScaler()
    X[num] = scaler.fit_transform(X[num])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    print('With KNN (K=1) accuracy is: ', knn.score(X_test, y_test))  # accuracy
    # Model complexity
    neig = np.arange(1, 25)
    train_accuracy = []
    test_accuracy = []
    # Loop over different values of k
    for i, k in enumerate(neig):
        # k from 1 to 25(exclude)
        knn = KNeighborsClassifier(n_neighbors=k)
        # Fit with knn
        knn.fit(X_train, y_train)
        # train accuracy
        train_accuracy.append(knn.score(X_train, y_train))
        # test accuracy
        test_accuracy.append(knn.score(X_test, y_test))

    # Plot
    plt.figure(figsize=[13, 8])
    plt.plot(neig, test_accuracy, label='Testing Accuracy')
    plt.plot(neig, train_accuracy, label='Training Accuracy')
    plt.legend()
    plt.title('-value VS Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.xticks(neig)
    plt.savefig('img/graph.png')
    plt.show()
    print("Best accuracy is {} with K = {}".format(np.max(test_accuracy), 1 + test_accuracy.index(np.max(test_accuracy))))


def linear(df):
    df = preprocessing.set_cat(df)
    num = ['area', 'rooms', 'offer_date', 'year_built', 'floor']
    cat = ['category']  # No need for coeff
    #df.offer_date = df.offer_date.map(dt.datetime.toordinal)
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
    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    #print('RMSE Cross-Validation:', np.sqrt(-cross_val_score(lm, X, y, scoring="neg_mean_squared_error", cv=10)).mean())
    plt.scatter(y_test, predictions, edgecolor='black')
    plt.show()
    sns.histplot((y_test - predictions), bins=50, kde=True, edgecolor="black", linewidth=1, color='blue')
    plt.show()



def max_leaf_nodes(df):
    df = preprocessing.set_cat(df)
    num = ['area', 'rooms', 'offer_date', 'year_built', 'floor']
    cat = ['category']  # No need for coeff
    #df.offer_date = df.offer_date.map(dt.datetime.toordinal)
    X = num + cat
    y = df['price']
    X = df[X]
    X = pd.get_dummies(X, 'district')  # Encoding cat features in dataset X using the One-Hot Encoding method
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for max_leaf_nodes in [5, 50, 500, 5000, 15000]:
        mae = get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, mae))


# ----------------------------------------------------------------------------------------------------------------------


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return (mae)


def random_forest(df):
    df = preprocessing.set_cat(df)
    num = ['area', 'rooms', 'offer_date', 'year_built', 'floor']
    cat = ['category']  # No need for coeff
    #df.offer_date = df.offer_date.map(dt.datetime.toordinal)
    X = num + cat
    y = df['price']
    X = df[X]
    X = pd.get_dummies(X, 'district')  # Encoding cat features in dataset X using the One-Hot Encoding method
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create a Random Forest Regressor
    reg = RandomForestRegressor()
    # Train the model using the training sets
    reg.fit(X_train, y_train)
    # Model prediction on train data
    y_pred = reg.predict(X_train)
    print('R^2:',metrics.r2_score(y_train, y_pred))
    print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
    print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
    print('MSE:',metrics.mean_squared_error(y_train, y_pred))
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    print('RMSE Cross-Validation:', np.sqrt(-cross_val_score(reg, X, y, scoring="neg_mean_squared_error", cv=10)).mean())
    # Visualizing the differences between actual prices and predicted values
    plt.scatter(y_train, y_pred)
    plt.xlabel("Prices")
    plt.ylabel("Predicted prices")
    plt.title("Prices vs Predicted prices")
    plt.show()


def XGBoost(df):
    df = preprocessing.set_cat(df)
    num = ['area', 'rooms', 'offer_date', 'year_built', 'floor']
    cat = ['category']  # No need for coeff
    #df.offer_date = df.offer_date.map(dt.datetime.toordinal)
    X = num + cat
    y = df['price']
    X = df[X]
    X = pd.get_dummies(X, 'district')  # Encoding cat features in dataset X using the One-Hot Encoding method
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #Create a XGBoost Regressor
    reg = XGBRegressor()
    # Train the model using the training sets
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_train)
    # Model Evaluation
    print('R^2:',metrics.r2_score(y_train, y_pred))
    print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
    print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
    print('MSE:',metrics.mean_squared_error(y_train, y_pred))
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    print('RMSE Cross-Validation:', np.sqrt(-cross_val_score(reg, X, y, scoring="neg_mean_squared_error", cv=10)).mean())
    # Visualizing the differences between actual prices and predicted values
    plt.scatter(y_train, y_pred)
    plt.xlabel("Prices")
    plt.ylabel("Predicted prices")
    plt.title("Prices vs Predicted prices")
    plt.show()


def SVM(df):
    df = preprocessing.set_cat(df)
    num = ['area', 'rooms', 'offer_date', 'year_built', 'floor']
    cat = ['category']  # No need for coeff
    #df.offer_date = df.offer_date.map(dt.datetime.toordinal)
    X = num + cat
    y = df['price']
    X = df[X]
    X = pd.get_dummies(X, 'district')  # Encoding cat features in dataset X using the One-Hot Encoding method
    scaler = StandardScaler()
    X[num] = scaler.fit_transform(X[num])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #Create a XGBoost Regressor
    reg = svm.SVR()
    # Train the model using the training sets
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_train)
    # Model Evaluation
    print('R^2:',metrics.r2_score(y_train, y_pred))
    print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
    print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
    print('MSE:',metrics.mean_squared_error(y_train, y_pred))
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    #print('RMSE Cross-Validation:', np.sqrt(-cross_val_score(reg, X, y, scoring="neg_mean_squared_error", cv=10)).mean())
    # Visualizing the differences between actual prices and predicted values
    plt.scatter(y_train, y_pred)
    plt.xlabel("Prices")
    plt.ylabel("Predicted prices")
    plt.title("Prices vs Predicted prices")
    plt.show()


def main():
    all_df = preprocessing.join_all_files()
    # ------------------------------------------------------------------------------------------------------------------
    # Cleaned and converted data
    # ------------------------------------------------------------------------------------------------------------------
    clean_df = preprocessing.clean_data(all_df)
    clean_df.offer_date = clean_df.offer_date.map(dt.datetime.toordinal)
    # ------------------------------------------------------------------------------------------------------------------
    # Correlation after removing outliers
    # ------------------------------------------------------------------------------------------------------------------
    out_of_outliers = preprocessing.outliers(clean_df)
    # ------------------------------------------------------------------------------------------------------------------
    # Visualization and prediction
    # ------------------------------------------------------------------------------------------------------------------
    clean_df = preprocessing.add_means(clean_df)  # not sure if needed
    # models_verification(clean_df) # old function
    # models_verification(out_of_outliers) # old function
    knn(clean_df)
    knn(out_of_outliers)
    max_leaf_nodes(clean_df)
    max_leaf_nodes(out_of_outliers)
    print("-" * 50)
    linear(clean_df)
    print("-" * 50)
    linear(out_of_outliers)
    print("-" * 50)
    random_forest(clean_df)
    print("-" * 50)
    random_forest(out_of_outliers)
    print("-" * 50)
    XGBoost(clean_df)
    print("-" * 50)
    XGBoost(out_of_outliers)
    print("-" * 50)
    
    print("-" * 50)
    SVM(clean_df)
    print("-" * 50)
    SVM(out_of_outliers)
    print("-" * 50)


if __name__ == '__main__':
    main()

