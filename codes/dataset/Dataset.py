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
    
    def make_windowed_dataset(self, 
                              df_X: pd.DataFrame,
                              df_y: pd.DataFrame,
                              window_size: int, 
                              forecast_horizon: int):
        """
        Create windowed dataset
        """
        X = []
        y = []
        for i in range(len(df_X) - window_size - forecast_horizon + 1):
            X.append(df_X.iloc[i:i+window_size].values)
            y.append(df_y.iloc[i+window_size:i+window_size+forecast_horizon].values)
        return np.array(X), np.array(y)
    
    def train_test_split(self,
                         df: pd.DataFrame,
                         station_name: str,
                         target: str,
                         window_size: int,
                         forecast_horizon: int,
                         granularity: str, 
                         test_ratio: float = 0.2
                        ):
        """
        Split the data into train and test sets
        """
        # select the station
        df = df[df['station'] == station_name].drop(columns='station')
        
        # if datetime is not the index, set it as index
        if df.index.name != 'datetime':
            df.set_index('datetime', inplace=True)
            # to_datetime
            df.index = pd.to_datetime(df.index)
        if granularity == 'hourly':
            df = df.resample('H').mean()
        elif granularity == 'daily':
            df = df.resample('D').mean()
        elif granularity == 'weekly':
            df = df.resample('W').mean()
        else:
            raise ValueError('Granularity should be either hourly, daily or weekly')
        
        df_X = df.drop(columns=target)
        df_y = df[target]

        # keep the last test_ratio% data for testing
        test_size = int(len(df) * test_ratio)
        train_size = len(df) - test_size
        df_X_train = df_X.iloc[:train_size]
        df_X_test = df_X.iloc[train_size:]

        df_y_train = df_y.iloc[:train_size]
        df_y_test = df_y.iloc[train_size:]

        # create the windowed dataset
        X_train, y_train = self.make_windowed_dataset(df_X_train, window_size, forecast_horizon)
        X_test, y_test = self.make_windowed_dataset(df_X_test, window_size, forecast_horizon)
        
        return X_train, y_train, X_test, y_test

  
if __name__ == "__main__":
    dataset = Dataset()
    df = dataset.process_data()

    station_name = 'Shunyi'
    target = 'PM2.5'
    window_size = 24
    forecast_horizon = 5
    granularity = 'weekly'
    test_ratio = 0.2


    dataset.train_test_split(df, station_name, target, window_size, forecast_horizon, granularity)


