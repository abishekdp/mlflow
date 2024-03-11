import json
import os
from pandas import read_csv, to_datetime
from matplotlib import pyplot as plt
import mlflow
from pmdarima import auto_arima, model_selection
from pmdarima.datasets import load_airpassengers
from pmdarima.utils import plot_forecast
from mlflow.models import infer_signature
import numpy as np

ARTIFACT_PATH = "model"


def calculate_cv_metrics(model, endog, metric, cv):
    cv_metric = model_selection.cross_val_score(model, endog, cv=cv, scoring=metric, verbose=0)
    return cv_metric[~np.isnan(cv_metric)].mean()


def save_and_log_plots(fig, filename):
    """Save a matplotlib figure to a file and log it as an artifact to MLflow."""
    fig.savefig(filename)
    mlflow.log_artifact(filename)


with mlflow.start_run():
    data = load_airpassengers()

    train_size = int(len(data) * 0.8)  # Use 80% of the data for training
    train, test = data[:train_size], data[train_size:]

    print("Training AutoARIMA model...")
    arima = auto_arima(
        train,
        error_action="ignore",
        trace=False,
        suppress_warnings=True,
        maxiter=5,
        seasonal=True,
        m=12,
    )

    print("Model trained. \nExtracting parameters...")
    parameters = arima.get_params(deep=True)

    metrics = {x: getattr(arima, x)() for x in ["aicc", "aic", "bic", "hqic", "oob"]}

    # Cross validation backtesting
    cross_validator = model_selection.RollingForecastCV(h=10, step=20, initial=60)

    for x in ["smape", "mean_absolute_error", "mean_squared_error"]:
        metrics[x] = calculate_cv_metrics(arima, data, x, cross_validator)

    print(f"Metrics: \n{json.dumps(metrics, indent=2)}")
    print(f"Parameters: \n{json.dumps(parameters, indent=2)}")

    predictions = arima.predict(n_periods=30, return_conf_int=False)
    signature = infer_signature(train, predictions)

    # Plot and save figures
    fig_components = arima.plot_diagnostics()
    save_and_log_plots(fig_components, "components_plot.png")

    fig_forecast = plt.figure()
    arima.plot_forecast(30, ax=fig_forecast.gca())
    save_and_log_plots(fig_forecast, "forecast_plot.png")

    fig_residuals = arima.plot_residuals()
    save_and_log_plots(fig_residuals, "residuals_plot.png")

    mlflow.pmdarima.log_model(
        pmdarima_model=arima, artifact_path=ARTIFACT_PATH, signature=signature
    )
    mlflow.log_params(parameters)
    mlflow.log_metrics(metrics)
    model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)

    print(f"Model artifact logged to: {model_uri}")

loaded_model = mlflow.pmdarima.load_model(model_uri)

forecast = loaded_model.predict(30)

print(f"Forecast: \n{forecast}")
