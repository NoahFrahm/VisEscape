import pandas as pd
import argparse

def compute_aggregate_metrics(df: pd.DataFrame) -> Dict:
    aggregate_metrics = {}
    metric_keys = ["success", "steps", "progress", "goal_completion_ratio", "spl"]
    
    for key in metric_keys:
        aggregate_metrics[f"{key}_mean"] = df[key].mean()
        aggregate_metrics[f"{key}_std"] = df[key].std()
        print(f"{key.capitalize()} - Mean: {aggregate_metrics[f'{key}_mean']:.4f}, Std: {aggregate_metrics[f'{key}_std']:.4f}")
    
    return aggregate_metrics



def main():
    parser = argparse.ArgumentParser(description='Get aggregate metrics from metrics csv')
    parser.add_argument('--metrics', type=str, required=True,
                        help='Path to the metrics CSV file')
    parser.add_argument('--out-file', required=True,
                        help='file path to save metrics to')
    
    args = parser.parse_args()
        
    
    
    df = pd.read_csv(args.metrics)
    compute_aggregate_metrics(df)
    
    # convert these into the summary metrics as seen in paper
    # - by room compute the deviation and averages for each metric
    # - aggregate across rooms to get overall averages and deviations

if __name__ == "__mani__":
    main()