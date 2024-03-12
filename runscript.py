import mlflow
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt

# Set the DagsHub tracking URI
remote_server_uri = "https://dagshub.com/abishekdp/mlflow.mlflow"
mlflow.set_tracking_uri(remote_server_uri)

# Load the model
model = mlflow.sklearn.load_model("runs:/a9d5f8a55d9c4226a9564543eee7bae8/prophet_model")

# Generate future dates
future = model.make_future_dataframe(periods=48, freq='MS')

# Make predictions
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.savefig('forecast_plot.png')
plt.close(fig)

# Perform cross-validation
initial = 2 * 365
initial = str(initial) + ' days'
period = 2 * 365
period = str(period) + ' days'
horizon = 365
horizon = str(horizon) + ' days'
cv_results = cross_validation(model, initial=initial, period=period, horizon=horizon)

# Calculate performance metrics
metrics = performance_metrics(cv_results)

# Save the metrics to a CSV file
metrics.to_csv('metrics.csv', index=False)

# Log the metrics and plots as artifacts with a specific run name
with mlflow.start_run(run_name='MyRunName'):
    mlflow.log_artifact('metrics.csv', 'metrics.csv')
    mlflow.log_artifact('forecast_plot.png', 'forecast_plot.png')

    # Log individual metric values
    for metric_name in metrics.columns:
        metric_value = metrics[metric_name].mean()
        if isinstance(metric_value, pd.Timedelta):
            # Convert Timedelta to seconds
            metric_value = metric_value.total_seconds()
        mlflow.log_metric(metric_name, metric_value)

