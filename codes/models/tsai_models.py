from tsai.tslearner import TSForecaster
from tsai.inference import load_learner
from tsai.metrics import mae, mse, mape
from tsai.callback.core import ShowGraph
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
import git

repoPath = git.Repo('.', search_parent_directories=True).working_tree_dir
model_savePath = os.path.join(repoPath, 'saved_tsai_models')
results_savePath = os.path.join(repoPath, 'results')


class TSAI_Models:
    def __init__(self) -> None:
        pass

    def train_model(self, **kwargs) -> None:
        """
        Train the model
        """
        # create a TSForecaster object
        X = kwargs.get('X')
        y = kwargs.get('y')
        splits = kwargs.get('splits')
        tfms = kwargs.get('tfms')
        batch_tfms = kwargs.get('batch_tfms')
        arch = kwargs.get('arch')
        epochs = kwargs.get('epochs')
        lr = kwargs.get('lr')
        bs = kwargs.get('batch_size')

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
         
        fcst = TSForecaster(X, y, splits=splits, path='models', tfms=tfms, 
                        batch_tfms=batch_tfms, bs=bs, arch=arch, metrics=mae)

        # train the model
        fcst.fit_one_cycle(epochs, lr)

        # save the model
        path = os.path.join(model_savePath, arch + '.pkl')
        fcst.export(path)

        return

    def evaluate_model(self, **kwargs) -> dict:
        """
        Evaluate the model
        """
        arch = kwargs.get('arch')
        #fcst = kwargs.get('fcst')
        X = kwargs.get('X')
        y = kwargs.get('y')
        splits = kwargs.get('splits')

        saved_model_path = os.path.join(model_savePath, arch + '.pkl')
        
        # load the model
        fcst = load_learner(saved_model_path)
        raw_preds, target, preds = fcst.get_X_preds(X[splits[1]], y[splits[1]])

        return raw_preds, target, preds
    
    def plot_predictions(self, archName, target, preds):
        # plot original and predicted time series
        target_flattened = target.flatten()
        preds_flattened = []
        for pred in preds:
            preds_flattened.extend(pred)
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=range(len(target_flattened)), y=target_flattened, label='Original')
        sns.lineplot(x=range(len(target_flattened)), y=preds_flattened, label='Forecast')
        plt.legend()
        #plt.show()

        # save the plot
        plt.savefig(os.path.join(results_savePath, archName + '.png'))

        return

        
