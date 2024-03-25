import os, sys
import pandas as pd
import numpy as np
import git

from tsai.data.validation import TimeSplitter
from tsai.data.preparation import SlidingWindow
from tsai.data.core import TSForecasting
from tsai.data.preprocessing import TSStandardize


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
        if os.path.exists(os.path.join(processed_data_folder, dataset_name + '_processed.parquet')):
            df = pd.read_parquet(os.path.join(processed_data_folder, dataset_name + '_processed.parquet'))
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
        # save as parquet
        df.to_parquet(os.path.join(processed_data_folder, dataset_name + '_processed.parquet'))
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
                         test_ratio: float = 0.2,
                         **kwargs
                        ):
        """
        Split the data into train and test sets
        """
        # select the station
        df = df[df['station'] == station_name].drop(columns='station')
    
        
        # get_year, get_month from kwargs
        if kwargs['get_year']:
            df['year'] = df.index.year
        if kwargs['get_month']:
            df['month'] = df.index.month

        # apply granularity
        df = self.apply_granularity(df, granularity)
        
        df_X = df.drop(columns=target)
        df_y = df[target]

        # keep the last test_ratio% data for testing
        test_size = int(len(df) * test_ratio)
        train_size = len(df) - test_size
        train_X, test_X = df_X.iloc[:train_size], df_X.iloc[train_size:]
        train_y, test_y = df_y.iloc[:train_size], df_y.iloc[train_size:]

        # create windowed dataset
        X_train, y_train = self.make_windowed_dataset(train_X, train_y, window_size, forecast_horizon)
        X_test, y_test = self.make_windowed_dataset(test_X, test_y, window_size, forecast_horizon)

        # if normalization is required, min-max normalize the data
        if kwargs['normalize']:
            X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
            y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min())
            X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
            y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min())

        return X_train, y_train, X_test, y_test
    
    def apply_granularity(self, df: pd.DataFrame, granularity: str):
        """
        Apply the granularity to the data
        """
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
        return df
    
    def get_tsai_data(self, **kwargs):
        """
        Get the data in the format required by tsai library
        """
        # load the processed data   
        df = self.process_data()

        # select the station
        df = df[df['station'] == kwargs['station_name']].drop(columns='station')

        # gset features and target columns
        target_col = kwargs['target']
        target_position = df.columns.get_loc(target_col)
        get_x = [i for i in range(len(df.columns)) if i != target_position]
        get_y = [target_position]

        # granularities, hourly, daily, weekly
        df = self.apply_granularity(df, kwargs['granularity'])

        # SlidingWindow
        ts = df.values
        X, y = SlidingWindow(window_len=kwargs['window_size'], horizon=kwargs['forecast_horizon'], get_x=get_x, get_y=get_y)(ts)

        # TimeSplitter
        splitter = TimeSplitter(valid_size=0.1, 
                                test_size=kwargs['test_ratio'], 
                                fcst_horizon=kwargs['forecast_horizon'])(y)
        tfms = [None, TSForecasting()]
        batch_tfms = TSStandardize()

        return X, y, splitter, tfms, batch_tfms
    


  
if __name__ == "__main__":
    dataset = Dataset()
    df = dataset.process_data()

    kwargs = {
        'target': 'PM10',
        'station_name': 'Shunyi',
        'window_size': 24,
        'forecast_horizon': 5,
        'granularity': 'daily',
        'test_ratio': 0.2,
        'get_year': True,
        'get_month': True,
    }

    # load tsai data
    dataset.get_tsai_data(**kwargs)

    station_name = 'Shunyi'
    target = 'PM2.5'
    window_size = 24
    forecast_horizon = 5
    granularity = 'weekly'
    test_ratio = 0.2


    #dataset.train_test_split(df, station_name, target, window_size, forecast_horizon, granularity, test_ratio)


