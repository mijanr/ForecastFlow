import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('colorblind')

def plot_predictions(archName, target, preds):
        """
        Plot the predictions
        """
        # plot original and predicted time series
        target_flattened = target.flatten()
        preds_flattened = preds.flatten()

        # plot 50 time steps
        time_steps = 100

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(target_flattened[:time_steps], label='Original')
        ax.plot(preds_flattened[:time_steps], label='Predicted')
        ax.set_title(f'Original vs Predicted time series for {archName}')
        ax.set_xlabel('Time steps: flattened')
        ax.set_ylabel('Values')
        ax.legend()

        plt.tight_layout()
        plt.close(fig)

        return fig