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

    # find the best model based on 'metrics.mae' for each 'params.model_name'
    df_best = all_runs.groupby('params.model_name').apply(lambda x: x.loc[x['metrics.mae'].idxmin()]).reset_index(drop=True)
    
    # make a dictionary of with key as 'params.model_name' and value as artifact_uri
    # artifact dict starts from mlruns directory, and ends with /original_vs_predicted.png
    artifact_dict = dict()
    for index, row in df_best.iterrows():
        # split the artifact_uri to get the path from mlruns directory
        row['artifact_uri'] = row['artifact_uri'].split('ForecastFlow/')[1]
        artifact_dict[row['params.model_name']] = row['artifact_uri'] + "/original_vs_predicted.png"
    
    # save the best model artifact_uri to a file
    with open(f"{basePath}/utils/best_results.txt", "w") as f:
        for key, value in artifact_dict.items():
            f.write(f"{key}: {value}\n")

    return
    

if __name__ == "__main__":
    get_best_run()