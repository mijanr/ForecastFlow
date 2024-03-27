import numpy as np
import matplotlib.pyplot as plt
import os, sys
import git
import joblib

import xgboost as xgb

from sklearn.multioutput import MultiOutputRegressor

repoPath = git.Repo('.', search_parent_directories=True).working_tree_dir
model_savePath = os.path.join(repoPath, 'saved_models')
results_savePath = os.path.join(repoPath, 'results')


class XGB:  
    def __init__(self) -> None:
        pass

    def train_model(self, **kwrgs):
        """
        Train the model
        """
        X_train = kwrgs.get('X_train')
        y_train = kwrgs.get('y_train')
        arch = kwrgs.get('arch')

        # flatten the features to make it (n_samples, n_features)
        X_train = X_train.reshape(X_train.shape[0], -1)
        
        # check if a model is already trained with the given name (arch)
        if os.path.exists(os.path.join(model_savePath, arch + '.pkl')):
            # say model already exists, ask if user wants to train again
            print(f'Model {arch} already exists. Do you want to train again?')
            choice = input('Enter y/n: ')
            if choice == 'n':
                print('You can directly evaluate the model using the evaluate_model method')
                return
            elif choice == 'y':
                pass
            else:
                print('Invalid choice. Exiting...')
                sys.exit(1)

        # create a multioutput regressor
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
        model = MultiOutputRegressor(xgb_model)

        # train the model
        model.fit(X_train, y_train)

        # save the model
        path = os.path.join(model_savePath, arch + '.pkl')

        joblib.dump(model, path)

        return


    def evaluate_model(self, **kwargs):
        """
        Evaluate the model
        """

        arch = kwargs.get('arch')
        X_test = kwargs.get('X_test')
        y_test = kwargs.get('y_test')

        # flatten the features to make it (n_samples, n_features)
        X_test = X_test.reshape(X_test.shape[0], -1)

        # load the model
        saved_model_path = os.path.join(model_savePath, arch + '.pkl')

        model = joblib.load(saved_model_path)

        # make predictions
        preds = model.predict(X_test)

        # calculate the error, e.g., mae, mse
        metrics = {
            'mae': np.mean(np.abs(y_test - preds)),
            'mse': np.mean((y_test - preds)**2)
        }
        
        outputDict = {
            'target': y_test,
            'preds': preds,
            'metrics': metrics
        }

        return outputDict

    def plot_predictions(self, archName, target, preds):
        """
        Plot the predictions
        """
        # plot original and predicted time series


        target_flattened = target.flatten()
        preds_flattened = preds.flatten()

        # plot 50 time steps
        time_steps = 100

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(target_flattened[:time_steps], label='Original')
        ax.plot(preds_flattened[:time_steps], label='Predicted')
        ax.set_title(f'Original vs Predicted time series for {archName}')
        ax.legend()

        plt.tight_layout()
        plt.close(fig)

        return fig
    

if __name__ == '__main__':
    pass