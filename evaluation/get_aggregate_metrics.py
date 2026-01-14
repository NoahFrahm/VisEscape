import pandas as pd
import argparse
import pprint


def compute_aggregate_metrics(df):
    aggregate_metrics = {}
    metric_keys = ["success", "steps", "progress", "goal_completion_ratio", "spl"]
    
    for key in metric_keys:
        success_rates = df.groupby('room')[key].mean()
        aggregate_metrics[f"{key}_mean"] = float(success_rates.mean())

    return aggregate_metrics


def main():
    parser = argparse.ArgumentParser(description='Get aggregate metrics from metrics csv')
    parser.add_argument('--metrics-csv', type=str, required=True,
                        help='Path to the metrics CSV file')
    parser.add_argument('--out-file', default='aggregate_metrics.csv',
                        help='file path to save metrics to')
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv)
    aggregate = compute_aggregate_metrics(df)

    pprint.pprint(aggregate)
    # breakpoint()
    
    # convert these into the summary metrics as seen in paper
    # - by room compute the deviation and averages for each metric
    # - aggregate across rooms to get overall averages and deviations

    # python -m evaluation.get_aggregate_metrics \
    # --metrics-csv results/VisEscaper/logs/test_run/Qwen/Qwen3-VL-8B-Instruct/vlm/no_hint/evaluation_metrics.csv

    # python -m evaluation.get_aggregate_metrics --metrics-csv results/VisEscaper/logs/test_run/Qwen/Qwen3-VL-32B-Instruct/vlm/no_hint/evaluation_metrics.csv
if __name__ == "__main__":
    main()