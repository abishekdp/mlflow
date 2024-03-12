import json
import os
from pandas import read_csv, to_datetime
from matplotlib import pyplot
import mlflow
from prophet import Prophet, serialize
from sklearn.metrics import mean_absolute_error
from prophet.diagnostics import cross_validation, performance_metrics
from mlflow.models import infer_signature
import numpy as np

SOURCE_DATA = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_air_passengers.csv"
ARTIFACT_PATH = "model"
np.random.seed(12345)


def load_data(path):
    """Load the dataset."""
    return read_csv(path)


def preprocess_data(df):
    """Preprocess the dataset."""
    df.columns = ['ds', 'y']
    df['ds'] = to_datetime(df['ds'])
    df['y'] = df['y'].astype(float)  # Convert 'y' column to float64
    return df


def extract_params(pr_model):
    """Extract parameters from the Prophet model."""
    return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}


def train_and_evaluate_model(df):
    """Train and evaluate the Prophet model."""
    model = Prophet()
    model.fit(df)

    metric_keys = ["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]
    metrics_raw = cross_validation(
        model=model,
        horizon="365 days",
        period="60 days",  # Adjusted period to reduce warnings
        initial="710 days",
        parallel="threads",
        disable_tqdm=True,
    )
    cv_metrics = performance_metrics(metrics_raw)
    metrics = {k: cv_metrics[k].mean() for k in metric_keys}

    train = model.history
    predictions = model.predict(model.make_future_dataframe(30))
    signature = infer_signature(train, predictions)

    remote_server_uri = "https://dagshub.com/abishekdp/mlflow.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)

    return model, predictions, metrics, signature


def log_mlflow(run_name, path, df, model, predictions, metrics, signature):
    """Log data, parameters, metrics, and artifacts to MLflow."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("data_path", path)
        mlflow.log_param("num_rows", df.shape[0])  # Log number of rows
        mlflow.log_param("num_cols", df.shape[1])  # Log number of columns
        mlflow.log_param("model_info", str(model))

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log Prophet model
        mlflow.prophet.log_model(model, artifact_path=ARTIFACT_PATH, signature=signature)

        # Plot components
        fig_components = model.plot_components(model.predict(df))
        components_plot_path = "components_plot.png"
        fig_components.savefig(components_plot_path)
        mlflow.log_artifact(components_plot_path)

        # Plot forecast
        fig_forecast = model.plot(predictions)
        forecast_plot_path = "forecast_plot.png"
        fig_forecast.savefig(forecast_plot_path)
        mlflow.log_artifact(forecast_plot_path)

        # Plot expected vs actual
        fig_expected_vs_actual = pyplot.plot(df['y'], label='Actual')
        pyplot.plot(predictions['yhat'], label='Predicted')
        pyplot.legend()
        expected_vs_actual_plot_path = "expected_vs_actual_plot.png"
        pyplot.savefig(expected_vs_actual_plot_path)
        mlflow.log_artifact(expected_vs_actual_plot_path)


def main():
    # Set the MLflow experiment
    mlflow.set_experiment("AirPassengers_Forecasting")

    # Define paths
    data_path = SOURCE_DATA
    run_name = "AirPassengers_Forecasting"

    try:
        # Load data
        df = load_data(data_path)

        # Preprocess data
        df = preprocess_data(df)

        # Train and evaluate model
        model, predictions, metrics, signature = train_and_evaluate_model(df)

        # Log to MLflow
        log_mlflow(run_name, data_path, df, model, predictions, metrics, signature)

        print("Experiment completed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        mlflow.log_param("error_message", str(e))


if __name__ == "__main__":
    main()

