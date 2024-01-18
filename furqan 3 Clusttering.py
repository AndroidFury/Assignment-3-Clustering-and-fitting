 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from lmfit import Model
import seaborn as sns
import matplotlib.font_manager as fm
from scipy import stats
def read_clean_transpose_csv(csv_file_path):
    """
    Read CSV file, handle missing values, and transpose the data.

    Parameters:
        csv_file_path (str): Path to the CSV file.

    Returns:
        tuple: Original data, cleaned data, and transposed data.
    """
    original_data = pd.read_csv(csv_file_path)
    original_data.replace('..', np.nan, inplace=True)

    columns_of_interest = [
        "Forest area (% of land area) [AG.LND.FRST.ZS]",
        "Adjusted net national income per capita (annual % growth) [NY.ADJ.NNTY.PC.KD.ZG]",
        "Agriculture, forestry, and fishing, value added (% of GDP) [NV.AGR.TOTL.ZS]",
        "Arable land (% of land area) [AG.LND.ARBL.ZS]"
    ]

    imputer = SimpleImputer(strategy='mean')
    cleaned_data = original_data.copy()
    cleaned_data[columns_of_interest] = imputer.fit_transform(cleaned_data[columns_of_interest])

    transposed_data = cleaned_data.transpose()

    return original_data, cleaned_data, transposed_data

def exponential_growth_model(x, amplitude, growth_rate):
    """
    Exponential growth model function.

    Parameters:
        x (array-like): Input data.
        amplitude (float): Amplitude parameter.
        growth_rate (float): Growth rate parameter.

    Returns:
        array-like: Model predictions.
    """
    return amplitude * np.exp(growth_rate * np.array(x))

def plot_curve_fit(time_data, actual_data, fitted_curve, uncertainty, future_years, predicted_values):
    """
    Plot the curve fit for the exponential growth model.

    Parameters:
        time_data (array-like): Time data.
        actual_data (array-like): Actual data.
        fitted_curve (array-like): Fitted curve.
        uncertainty (array-like): Uncertainty in the fitted curve.
        future_years (list): List of future years for prediction.
        predicted_values (array-like): Predicted values for future years.
    """
    custom_font = fm.FontProperties(fname=fm.findfont(fm.FontProperties(family='arial')))
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    sns.scatterplot(x=time_data, y=actual_data, label='Actual Data', color='#3498db', alpha=0.7, s=80)
    sns.lineplot(x=time_data, y=fitted_curve, label='Exponential Growth Fit', color='#2ecc71', linewidth=2)

    plt.fill_between(time_data, fitted_curve - uncertainty,
                     fitted_curve + uncertainty,
                     color='#2ecc71', alpha=0.2, label='95% Confidence Interval')

    plt.xlabel('Time', fontproperties=custom_font, fontsize=14)
    plt.ylabel('Forest area (% of land area) [AG.LND.FRST.ZS]', fontproperties=custom_font, fontsize=14)
    plt.title('Curve Fit for Forest Area Over Time', fontproperties=custom_font, fontsize=16)

    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, framealpha=0.8, prop=custom_font)
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.gca().set_facecolor('#ecf0f1')
    plt.tick_params(axis='both', colors='#2c3e50')

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#2c3e50')

    plt.show()

def plot_prediction(time_data, actual_data, fitted_curve, uncertainty, future_years, predicted_values):
    """
    Plot the curve fit and prediction for future years.

    Parameters:
        time_data (array-like): Time data.
        actual_data (array-like): Actual data.
        fitted_curve (array-like): Fitted curve.
        uncertainty (array-like): Uncertainty in the fitted curve.
        future_years (list): List of future years for prediction.
        predicted_values (array-like): Predicted values for future years.
    """
    df = pd.DataFrame({'Time': time_data, 'Actual Data': actual_data,
                       'Exponential Growth Fit': fitted_curve,
                       '95% Confidence Interval': uncertainty})

    df_melted = pd.melt(df, id_vars='Time', var_name='Variable', value_name='Value')

    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 8))
    plt.ylim(0, 100)

    sns.lineplot(x='Time', y='Value', hue='Variable', data=df_melted, markers=True)
    plt.scatter(future_years, predicted_values, color='red', label='Predicted Values')

    plt.xlabel('Time')
    plt.ylabel('Arable land (% of land area) [AG.LND.ARBL.ZS]')
    plt.title('Curve Fit and Prediction for Arable Land Over Time')

    plt.legend()
    plt.show()

def cluster_and_visualize(original_data, cleaned_data):
    """
    Perform clustering and visualize the results.

    Parameters:
        original_data (pd.DataFrame): Original data.
        cleaned_data (pd.DataFrame): Cleaned data.
    """
    columns_for_clustering = [
        "Forest area (% of land area) [AG.LND.FRST.ZS]",
        "Adjusted net national income per capita (annual % growth) [NY.ADJ.NNTY.PC.KD.ZG]",
        "Agriculture, forestry, and fishing, value added (% of GDP) [NV.AGR.TOTL.ZS]",
        "Arable land (% of land area) [AG.LND.ARBL.ZS]"
    ]

    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(cleaned_data[columns_for_clustering])

    kmeans = KMeans(n_clusters=3, random_state=42, init='k-means++', max_iter=300)
    cleaned_data['Cluster'] = kmeans.fit_predict(df_normalized)

    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    silhouette_avg = silhouette_score(df_normalized, cleaned_data['Cluster'])
    print(f"Silhouette Score: {silhouette_avg:.4f}")

    sns.set(style="whitegrid")
    plt.scatter(cleaned_data[columns_for_clustering[1]],
                cleaned_data[columns_for_clustering[3]],
                c=cleaned_data['Cluster'], cmap='viridis', s=50, edgecolor='black', linewidth=0.5, alpha=0.7,
                label='Data Points')

    plt.scatter(cluster_centers[:, 1], cluster_centers[:, 3],
                marker='X', s=200, c='red', edgecolor='black', linewidth=1.5, label='Cluster Centers')

    plt.title('Clustering of Countries with Cluster Centers', fontsize=16)
    plt.xlabel('Adjusted Net National Income Growth (%)', fontsize=12)
    plt.ylabel('Arable Land (% of Land Area)', fontsize=12)

    cbar = plt.colorbar()
    cbar.set_label('Cluster', rotation=270, labelpad=15)

    plt.legend()
    plt.show()

def fit_exponential_growth_model(time_points, actual_values):
    """
    Fits an exponential growth model to the provided time points and actual values.

    Parameters:
    - time_points (array-like): Time points.
    - actual_values (array-like): Actual data values.

    Returns:
    - fitting_result (lmfit.model.ModelResult): Result of the curve fitting.
    """

    # Define the exponential growth model function
    def exponential_growth_model(x, amplitude, growth_rate):
        return amplitude * np.exp(growth_rate * np.array(x))

    # Create a model and set initial parameters
    model = Model(exponential_growth_model)
    initial_params = model.make_params(amplitude=1, growth_rate=0.001)

    # Fit the model to the data
    fitting_result = model.fit(actual_values, x=time_points, params=initial_params)

    return fitting_result

def err_ranges(x, params, covariance, conf=0.95):
    """
    Calculate the confidence interval for the curve fit parameters.

    Parameters:
    - x (array-like): Independent variable.
    - params (lmfit.Parameters): Parameters of the model.
    - covariance (array-like): Covariance matrix of the parameters.
    - conf (float, optional): Confidence level. Default is 0.95.

    Returns:
    - tuple: Lower and upper bounds of the confidence interval.
    """
    perr = np.sqrt(np.diag(covariance))
    alpha = 1 - conf
    nstd = stats.norm.ppf(1 - alpha / 2)

    # Extract parameter values
    amplitude, growth_rate = params['amplitude'].value, params['growth_rate'].value

    # Calculate lower and upper bounds
    lower = exponential_growth_model(x, amplitude - nstd * perr[0], growth_rate - nstd * perr[1])
    upper = exponential_growth_model(x, amplitude + nstd * perr[0], growth_rate + nstd * perr[1])

    return lower, upper
def plot_confidence_interval(time_points, fitted_curve, confidence_interval,actual_data):
    """
    Plot the curve fit with confidence interval.

    Parameters:
    - time_points (array-like): Time points.
    - fitted_curve (array-like): Fitted curve.
    - confidence_interval (tuple): Lower and upper bounds of the confidence interval.
    """
    custom_font = fm.FontProperties(fname=fm.findfont(fm.FontProperties(family='arial')))
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    sns.scatterplot(x=time_points, y=actual_data, label='Actual Data', color='#3498db', alpha=0.7, s=80)
    sns.lineplot(x=time_points, y=fitted_curve, label='Exponential Growth Fit', color='#2ecc71', linewidth=2)

    # Plot confidence interval
    plt.fill_between(time_points, confidence_interval[0], confidence_interval[1],
                     color='#2ecc71', alpha=0.2, label='95% Confidence Interval')

    plt.xlabel('Time', fontproperties=custom_font, fontsize=14)
    plt.ylabel('Arable land (% of land area) [AG.LND.ARBL.ZS]', fontproperties=custom_font, fontsize=14)
    plt.title('Curve Fit with Confidence Interval for Arable Land Over Time', fontproperties=custom_font, fontsize=16)

    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, framealpha=0.8, prop=custom_font)
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.gca().set_facecolor('#ecf0f1')
    plt.tick_params(axis='both', colors='#2c3e50')

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#2c3e50')

    plt.show()
def main():
    # Specify the path to the CSV file
    csv_file_path = 'forestdata3.csv'

   # Read, clean, and transpose the data
    original_data, cleaned_data, transposed_data = read_clean_transpose_csv(csv_file_path)

   # Perform clustering and visualize
    cluster_and_visualize(original_data, cleaned_data)


    # Extract relevant time and actual data for curve fitting
    time_data = cleaned_data['Time']
    actual_data = cleaned_data['Arable land (% of land area) [AG.LND.ARBL.ZS]']

    # Fit exponential growth model
    fitting_result = fit_exponential_growth_model(time_data, actual_data)

# Calculate confidence interval
    confidence_interval = err_ranges(time_data, fitting_result.params, fitting_result.covar)

# Specify future years for prediction
    future_years = [2025, 2030, 2035]

# Predict values for future years and plot curve fit
    predicted_values = fitting_result.eval(x=np.array(future_years))
    plot_curve_fit(time_data, actual_data, fitting_result.best_fit, fitting_result.eval_uncertainty(), future_years, predicted_values)

# Print predicted values for future years
    for year, value in zip(future_years, predicted_values):
        print(f"Predicted value for {year}: {value:.2f}")

# Plot curve fit, prediction, and confidence interval for arable land over time
    plot_prediction(time_data, actual_data, fitting_result.best_fit, fitting_result.eval_uncertainty(), future_years, predicted_values)
    plot_confidence_interval(time_data, fitting_result.best_fit, confidence_interval,actual_data)

if __name__ == "__main__":
    main()
