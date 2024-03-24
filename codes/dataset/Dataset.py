import os, sys
import pandas as pd
import numpy as np
import git


class Dataset:
    def __init__(self) -> None:
        self.base_path = git.Repo('.', search_parent_directories=True).working_tree_dir
        self.dataset_name = 'Beijing_AirQuality'

    def load_data(self) -> pd.DataFrame:
        """
        Load the raw data
        """        
        data_folder = os.path.join(self.base_path, 'data','raw', 'Beijing_AirQuality')
        files = os.listdir(data_folder)
        dfs = []
        for file in files:
            df = pd.read_csv(os.path.join(data_folder, file), encoding='utf-8', parse_dates=['year'])
            dfs.append(df)
        df = pd.concat(dfs)

        # drop 'No' column
        df.drop(columns='No', inplace=True)
        return df
    def process_data(self) -> pd.DataFrame:
        """
        Process the raw data
        """
        dataset_name = self.dataset_name

        # if already processed, load the processed data
        processed_data_folder = os.path.join(self.base_path, 'data', 'processed')
        if os.path.exists(os.path.join(processed_data_folder, dataset_name + '_processed.csv')):
            df = pd.read_csv(os.path.join(processed_data_folder, dataset_name + '_processed.csv'))
            return df
        else:
            df = self.load_data()
            
            # ffill missing values
            df.ffill(inplace=True)

            # drop extra infromation from year column
            df['year'] = df['year'].dt.year

            # now combine year, month, day, hour columns to create a datetime column
            df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

            # drop year, month, day, hour columns
            df.drop(columns=['year', 'month', 'day', 'hour'], inplace=True)

            # set datetime as index
            df.set_index('datetime', inplace=True)

            # drop wind direction column
            df.drop(columns='wd', inplace=True)

        # if not saved already, save the processed data
        df.to_csv(os.path.join(processed_data_folder, dataset_name + '_processed.csv'))
        return df
    
    def make_windowed_dataset(self, df: pd.DataFrame, window_size: int, forecast_horizon: int):
        """
        Create windowed dataset
        """
        X = []
        y = []
        for i in range(len(df) - window_size - forecast_horizon + 1):
            X.append(df.iloc[i:i+window_size].values)
            y.append(df.iloc[i+window_size:i+window_size+forecast_horizon].values)
        X = np.array(X)
        y = np.array(y)
        return X, y
    
if __name__ == "__main__":
    dataset = Dataset()
    df = dataset.process_data()

