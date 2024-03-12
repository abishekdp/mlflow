import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.diagnostics import cross_validation, performance_metrics
import mlflow
import mlflow.sklearn

# Load the data
data_path = 'AirPassengers.csv'
run_name = 'AirPassengers_Forecasting'

remote_server_uri = "https://dagshub.com/abishekdp/mlflow.mlflow"
mlflow.set_tracking_uri(remote_server_uri)

try:
    with mlflow.start_run(run_name=run_name):
        # Load data
        fb = pd.read_csv(data_path)
        fb.columns = ['ds', 'y']
        fb.dropna(inplace=True)
        fb['ds'] = pd.to_datetime(fb['ds'])

        # Split the data into training and test sets
        train = fb[:132]
        test = fb[132:]

        # Initialize Prophet model
        m = Prophet(seasonality_mode='multiplicative')
        m.fit(train)

        # Log parameters
        mlflow.log_param('seasonality_mode', 'multiplicative')

        # Make future dataframe for predictions
        future = m.make_future_dataframe(periods=12, freq='MS')
        forecast = m.predict(future)

        # Log plot artifacts
        fig = m.plot(forecast, uncertainty=True)
        fig_path = 'forecast_uncertainty.png'
        fig.savefig(fig_path)
        mlflow.log_artifact(fig_path)
        plt.close(fig)

        fig = m.plot(forecast)
        fig_path = 'forecast.png'
        fig.savefig(fig_path)
        mlflow.log_artifact(fig_path)
        plt.close(fig)

        ax = forecast.plot(x='ds', y='yhat', legend=True, label='Predictions', figsize=(12, 8))
        test.plot(x='ds', y='y', legend=True, label='True Test Data', ax=ax, xlim=('1960-01-01', '1961-01-01'))
        ax_path = 'forecast_test.png'
        ax.figure.savefig(ax_path)
        mlflow.log_artifact(ax_path)
        plt.close(ax.figure)

        fig = m.plot_components(forecast)
        components_path = 'components.png'
        fig.savefig(components_path)
        mlflow.log_artifact(components_path)
        plt.close(fig)

        # Perform cross-validation
        initial = 2 * 365
        initial = str(initial) + ' days'
        period = 2 * 365
        period = str(period) + ' days'
        horizon = 365
        horizon = str(horizon) + ' days'
        fb_cv = cross_validation(m, initial=initial, period=period, horizon=horizon)

        # Log performance metrics
        mlflow.log_metric('cv_rmse', performance_metrics(fb_cv)['rmse'].mean())
        mlflow.log_metric('rmse', performance_metrics(fb_cv)['rmse'].mean())

        # Log the trained model
        mlflow.sklearn.log_model(m, 'prophet_model')

        print("Experiment completed successfully!")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    mlflow.log_param("error_message", str(e))
