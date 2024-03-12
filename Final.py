import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import mlflow
import mlflow.sklearn
import pickle

# Set MLflow tracking URI
remote_server_uri = "https://dagshub.com/abishekdp/mlflow.mlflow"
mlflow.set_tracking_uri(remote_server_uri)

# Load the data
fb = pd.read_csv('AirPassengers.csv')
fb.columns = ['ds', 'y']
fb.dropna(inplace=True)
fb['ds'] = pd.to_datetime(fb['ds'])

# Split the data into training and test sets
train = fb[:132]
test = fb[132:]

# Initialize Prophet model
m = Prophet(seasonality_mode='multiplicative')
m.fit(train)


def log_plots_and_metrics(m, forecast, fb_cv, test):
    # Log plot artifacts
    fig = m.plot(forecast, uncertainty=True)
    mlflow.log_figure(fig, 'forecast_uncertainty.png')
    plt.close(fig)

    fig = m.plot(forecast)
    mlflow.log_figure(fig, 'forecast.png')
    plt.close(fig)

    ax = forecast.plot(x='ds', y='yhat', legend=True, label='Predictions', figsize=(12, 8))
    test.plot(x='ds', y='y', legend=True, label='True Test Data', ax=ax, xlim=('1960-01-01', '1961-01-01'))
    mlflow.log_figure(ax.figure, 'forecast_test.png')
    plt.close(ax.figure)

    fig = m.plot_components(forecast)
    mlflow.log_figure(fig, 'components.png')
    plt.close(fig)

    # Log performance metrics
    metrics = performance_metrics(fb_cv)
    for metric_name in metrics.columns:
        if metrics[metric_name].dtype == 'timedelta64[ns]':
            value = metrics[metric_name].mean().total_seconds()
        else:
            value = metrics[metric_name].mean()
        mlflow.log_metric(f'cv_{metric_name}', value)

    # Log the trained model
    mlflow.sklearn.log_model(m, 'prophet_model')

    # Save the Prophet model
    with open('prophet_model.pkl', 'wb') as f:
        pickle.dump(m, f)


# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param('seasonality_mode', 'multiplicative')

    # Make future dataframe for predictions
    future = m.make_future_dataframe(periods=12, freq='MS')
    forecast = m.predict(future)

    # Perform cross-validation
    initial = 2 * 365
    initial = str(initial) + ' days'
    period = 2 * 365
    period = str(period) + ' days'
    horizon = 365
    horizon = str(horizon) + ' days'
    fb_cv = cross_validation(m, initial=initial, period=period, horizon=horizon)

    # Log plots and metrics
    log_plots_and_metrics(m, forecast, fb_cv, test)
