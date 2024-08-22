import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import FunctionTransformer
import math

class DataCleaningAndVisualization:
    def data_info(self, data):
        cols = data.columns
        unique_val = [data[col].value_counts().head(10).index.to_numpy() for col in cols]
        n_uniques = [data[col].nunique() for col in cols]
        dtypes = [data[col].dtype for col in cols]
        nulss = [data[col].isnull().sum() for col in cols]
        dup = [data.duplicated().sum() for col in cols]
        return pd.DataFrame({'Col': cols, 'dtype': dtypes, 'n_uniques': n_uniques, 'n_nan': nulss, 'unique_val': unique_val, 'duplicated': dup})

    def categorical_plot(self, data, columns, numbers_of_cat_value):
        total_cols = 3
        total_rows = math.ceil(len(columns) / total_cols)
        plt.figure(figsize=(20, 5 * total_rows))
        plot_idx = 1
        for column in columns:
            if (data[column].nunique()) <= numbers_of_cat_value:
                ax = plt.subplot(total_rows, total_cols, plot_idx)
                sns.countplot(x=data[column], palette="rocket", hue=data[column])
                ax.set_title(f"Count Plot of {column}")
                plot_idx += 1
            else:
                print(f"Column '{column}' is not categorical.")
        plt.tight_layout()
        plt.show()

    def continuous_plot(self, data, columns, numbers_of_cat_value):
        total_cols = 3
        total_rows = math.ceil(len(columns) / total_cols)
        plt.figure(figsize=(20, 5 * total_rows))
        plot_idx = 1
        for column in columns:
            if (data[column].nunique()) >= numbers_of_cat_value and (data[column].dtype != 'object') and (data[column].value_counts().iloc[0]>=numbers_of_cat_value):
                ax = plt.subplot(total_rows, total_cols, plot_idx)
                sns.histplot(data[column])
                ax.set_title(f"histograme plot of {column}")
                plot_idx += 1
            else:
                print(f"Column '{column}' is not countinuous.")
        plt.tight_layout()
        plt.show()

    def boxplot_numeric_columns(self, data):
        numeric_columns = data.select_dtypes(include=['number'])
        num_cols = len(numeric_columns.columns)
        plt.figure(figsize=(25, 5))
        for i, column in enumerate(numeric_columns.columns):
            plt.subplot(1, num_cols, i+1)
            sns.boxplot(x=numeric_columns[column])
            plt.title(f'Box plot for {column}')

    def drop_duplicat(self, data, columns_uniques):
        index=data[data.drop(columns_uniques,axis=1).duplicated()].index
        print("Number of duplicated rows is",len(index))
        return data.drop(index,axis=0)

    def remove_outliers_iqr_countinuous(self, data, columns):
        data_copy = data.copy()
        for col in columns:
            if data_copy[col].dtype != 'object':
                q1, q3 = data_copy[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_whisker, upper_whisker = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                filt = (data_copy[col] < lower_whisker) | (data_copy[col] > upper_whisker)
                data_copy = data_copy[~filt]
        return data_copy.reset_index(drop=True)

    def remove_outliers_z_score_countinuous(self, data, upper_threshold=3, lower_threshold=-3):
        data_copy = data.copy()
        feature = data_copy.select_dtypes(np.number).columns
        skewness = data_copy[feature].skew()
        for column in feature:
            if -0.5 <= skewness[column] <= 0.5:  
                z_scores = (data_copy[column] - data_copy[column].mean()) / data_copy[column].std()
                outliers = (z_scores < lower_threshold) | (z_scores > upper_threshold)
                data_copy = data_copy[~outliers]
        return data_copy.reset_index(drop=True)

    def Skewness_log_square(self, data, numbers_of_cat_value):
        features = data.select_dtypes(np.number).columns
        for feature in features:
            if data[feature].nunique() <= numbers_of_cat_value:  # Check if unique values are greater than 10
                print(f" Categorical Features {feature} ")
                continue

            skewness = data[feature].skew()
            if skewness > 0.5:
                transformation = np.log1p
                transformation_name = "Log"
            elif skewness < -0.5:
                transformation = np.square
                transformation_name = "Square"
            else:
                print(f"Normal distribution {feature}")
                continue

            plt.figure(figsize=(15, 6))
            plt.subplot(1, 2, 1)
            plt.title(f"Distribution of {feature} before Transformation", fontsize=15)
            sns.histplot(data[feature], kde=True, color="red")
            plt.subplot(1, 2, 2)

            df_transformed = transformation(data[feature])
            plt.title(f"Distribution of {feature} after Transformation", fontsize=15)
            sns.histplot(df_transformed, bins=20, kde=True, legend=False)
            plt.xlabel(feature)
            plt.show()

            print(f"Skewness was {round(data[feature].skew(), 5)} before & is {round(pd.Series(df_transformed).skew(), 5)} after {transformation_name} transformation.")
            data[feature] = df_transformed
        return data

    def remove_categorical_outliers(self, data, columns, threshold):
        for col in columns:
            series = data[col].value_counts()
            outliers = series[series < threshold].index
            data = data[~data[col].isin(outliers)]
            print(col, np.array(outliers))
        return data.reset_index(drop=True)
