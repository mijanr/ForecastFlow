import mlflow
import git

basePath = git.Repo('.', search_parent_directories=True).working_tree_dir
mlflow.set_tracking_uri(f"file:{basePath}/mlruns")


def get_best_run():
    # get all runs
    all_runs = mlflow.search_runs(search_all_experiments=True)

    # important columns

    columns = ['run_id', 'artifact_uri', 'metrics.mae', 'metrics.mse', 
           'params.model_name', 'params.granularity', 'params.window_size', 
           'params.target_column', 'params.horizon', 'tags.mlflow.runName']
    
    # filter only the columns we need
    all_runs = all_runs[columns]

    # for all 'params.model_name', find the best model based on 'metrics.mae', 
    #and make a dataframe 
    df_best = all_runs.groupby('params.model_name').apply(lambda x: 
                x.loc[x['metrics.mae'].idxmin()]).reset_index(drop=True)
    
    

    return
    

if __name__ == "__main__":
    get_best_run()