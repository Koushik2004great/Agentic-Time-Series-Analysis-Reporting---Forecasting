import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tsa.stattools import acf, pacf
from sklearn.linear_model import LinearRegression
from langchain_google_genai import GoogleGenerativeAI
import itertools
import ast
import warnings
import pickle
import json
import os

warnings.filterwarnings("ignore")

def _set_plot_style():
    """Sets a consistent and professional style for all plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (14, 7)
    plt.rcParams['font.size'] = 12

def list_csv_files_in_directory(directory: str) -> str:
    """TOOL: Lists all the .csv files in a specified directory."""
    print(f"AGENT ACTION: Listing CSV files in '{directory}'...")
    try:
        files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        if not files:
            return "No CSV files found in the directory."
        return json.dumps({"csv_files": files})
    except Exception as e:
        return f"Error listing files: {e}"

def get_csv_column_details(filepath: str) -> str:
    """TOOL: Inspects a CSV file and returns a list of its column names and their data types."""
    print(f"AGENT ACTION: Getting column details from '{filepath}'...")
    try:
        df = pd.read_csv(filepath, nrows=5) 
        return json.dumps({
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict()
        })
    except Exception as e:
        return f"Error inspecting CSV: {e}"

def ask_user_for_choice(query: str) -> str:
    """TOOL: Pauses and asks the user to choose from a list of options."""
    params = json.loads(query); question, options = params['question'], params['options']
    print(f"\n--- HUMAN INPUT REQUIRED ---\nAGENT: {question}")
    for i, option in enumerate(options): print(f"  [{i+1}] {option}")
    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(options):
                chosen_option = options[choice - 1]
                print(f"USER: You have selected '{chosen_option}'. Resuming analysis...")
                return f"The user chose to proceed with '{chosen_option}'."
            else: print("Invalid number. Please try again.")
        except ValueError: print("Invalid input. Please enter a number.")

def ask_user_for_text_input(question: str) -> str:
    """TOOL: Pauses and asks the user an open-ended question, returning their text response."""
    print(f"\n--- HUMAN INPUT REQUIRED ---\nAGENT: {question}")
    response = input("Your response: ")
    print(f"USER: You entered '{response}'. Resuming analysis...")
    return f"The user provided the following input: {response}"

def find_and_ask_categorical_filter(filepath: str) -> str:
    """
    TOOL: Scans a CSV file for a suitable categorical column for filtering. If exactly one
    is found, it automatically asks the user to choose a value from that column.
    """
    print(f"AGENT ACTION: Finding and asking for a categorical filter in '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
        categorical_cols = {}
        for col in df.select_dtypes(include=['object', 'category']).columns:
            unique_count = df[col].nunique()
            if 1 < unique_count <= 20:
                categorical_cols[col] = df[col].unique().tolist()
        
        if len(categorical_cols) == 1:
            column_name = list(categorical_cols.keys())[0]
            options = categorical_cols[column_name]
            
            print(f"\n--- HUMAN INPUT REQUIRED ---")
            print(f"AGENT: I've identified a filterable column: '{column_name}'. Which category would you like to analyze?")
            for i, option in enumerate(options):
                print(f"  [{i+1}] {option}")
            
            while True:
                try:
                    choice = int(input("Enter the number of your choice: "))
                    if 1 <= choice <= len(options):
                        chosen_option = options[choice - 1]
                        print(f"USER: You have selected '{chosen_option}'. Resuming analysis...")
                        return json.dumps({
                            "status": "filter_selected",
                            "category_col": column_name,
                            "category_to_filter": chosen_option
                        })
                    else:
                        print("Invalid number. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        elif len(categorical_cols) > 1:
            return json.dumps({"status": "too_many_categories", "message": f"Multiple potential categorical columns found: {list(categorical_cols.keys())}. Agent should ask user to specify which one to use."})
        else:
            return json.dumps({"status": "no_categories_found", "message": "No suitable categorical columns were found for filtering."})

    except Exception as e:
        return json.dumps({"status": "error", "message": f"An error occurred: {e}"})

def load_and_prepare_data(params: dict, temp_dir: str) -> str:
    """
    TOOL: Loads data, intelligently handles resampling, validates, filters, and saves the
    prepared time series to a temporary file, returning the filepath.
    """
    filepath = params['filepath']
    date_col = params['date_col']
    value_col = params['value_col']
    category_col = params.get('category_col')
    category_to_filter = params.get('category_to_filter')
    
    print(f"AGENT ACTION: Loading and preparing data from '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
        required_cols = [date_col, value_col]
        if category_col: required_cols.append(category_col)
        for col in required_cols:
            if col not in df.columns: raise ValueError(f"Column '{col}' not in file.")
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            df.dropna(subset=[value_col], inplace=True)
        if not pd.api.types.is_numeric_dtype(df[value_col]):
             raise TypeError(f"Value column '{value_col}' could not be converted to numeric.")
        df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True)
        
        if category_col and category_to_filter:
            print(f"...filtering by '{category_col}' == '{category_to_filter}'")
            df = df[df[category_col] == category_to_filter].copy()
            if df.empty: raise ValueError(f"No data found for category '{category_to_filter}'.")

        df.set_index(date_col, inplace=True); df.sort_index(inplace=True)

        last_raw_date = df.index.max()

        monthly_series = df[value_col].resample('MS').sum()

        if not last_raw_date.is_month_end:
            print(f"...Warning: Last data point is on {last_raw_date.date()}, which is not the month's end. Dropping the incomplete final month to ensure model accuracy.")
            monthly_series = monthly_series.iloc[:-1]
        
        if monthly_series.empty:
            raise ValueError("The final time series is empty after processing.")
        os.makedirs(temp_dir, exist_ok=True)
        temp_filepath = os.path.join(temp_dir, "current_timeseries.json")
        monthly_series.to_json(temp_filepath, date_format='iso')
        
        return json.dumps({
            "status": "Success",
            "message": "Data prepared, validated, and saved to temporary file.",
            "temp_timeseries_path": temp_filepath
        })
    except Exception as e:
        return json.dumps({"status": "Failure", "message": f"An error occurred in data preparation: {e}"})
    
# === TOOL 2: Data Quality & EDA ===
def check_for_missing_values(timeseries_json: str) -> str:
    """
    TOOL 2: Checks for any missing values (nulls) in the time series data.
    
    Args:
        timeseries_json (str): A JSON string of the time series from the previous step.

    Returns:
        str: A JSON string confirming the number of missing values found.
    """
    print("AGENT ACTION: Executing 'check_for_missing_values'...")
    s = pd.read_json(timeseries_json, typ='series')
    missing_count = int(s.isnull().sum())
    message = f"Warning: Found {missing_count} missing values." if missing_count > 0 else "No missing values found."
    
    return json.dumps({
        "status": "Success",
        "missing_value_count": missing_count,
        "message": message
    })
def perform_eda_and_save_plots(temp_filepath: str, plot_dir: str) -> str:
    """TOOL: Takes a filepath to a JSON time series, loads it, and performs EDA."""
    print("AGENT ACTION: Executing 'perform_eda_and_save_plots'...")
    try:
        s = pd.read_json(temp_filepath, typ='series')
        _set_plot_style()
        os.makedirs(plot_dir, exist_ok=True)
        filenames = []

        plt.figure(); s.plot(title='Monthly Sales Time Series'); plt.savefig(os.path.join(plot_dir, "01_time_series_plot.png")); plt.close(); filenames.append("01_time_series_plot.png")

        plt.figure(); sns.histplot(s, kde=True); plt.title('Distribution of Monthly Sales Values'); plt.savefig(os.path.join(plot_dir, "02_distribution_plot.png")); plt.close(); filenames.append("02_distribution_plot.png")

        df_plot = pd.DataFrame({'Sales': s})
        df_plot['Year'] = s.index.year
        df_plot['Month'] = s.index.month_name()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        df_plot['Month'] = pd.Categorical(df_plot['Month'], categories=month_order, ordered=True)

        plt.figure(); sns.lineplot(data=df_plot, x='Month', y='Sales', hue='Year', palette='viridis', marker='o'); plt.title('Seasonal Plot'); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "03_seasonal_plot.png")); plt.close(); filenames.append("03_seasonal_plot.png")

        fig, axes = plt.subplots(4, 3, figsize=(18, 15), sharey=True)
        axes = axes.flatten()
        for i, month in enumerate(month_order):
            month_data = df_plot[df_plot['Month'] == month]
            axes[i].plot(month_data['Year'], month_data['Sales'], marker='o')
            axes[i].hlines(month_data['Sales'].mean(), xmin=df_plot['Year'].min(), xmax=df_plot['Year'].max(), color='gray', linestyle='--')
            axes[i].set_title(month)
            axes[i].set_xlabel('Year')
        fig.suptitle('Seasonal Subseries Plot', fontsize=20, y=1.02); plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "04_seasonal_subseries_plot.png")); plt.close(); filenames.append("04_seasonal_subseries_plot.png")

        X = np.arange(len(s)).reshape(-1, 1); y = s.values
        model = LinearRegression(); model.fit(X, y)
        trend_slope = model.coef_[0]
        trend_interpretation = "strong upward trend" if trend_slope > 100 else "slight upward trend" if trend_slope > 0 else "downward trend"

        return json.dumps({
            "status": "Success",
            "message": f"Comprehensive EDA complete. {len(filenames)} plots saved.",
            "plot_filenames": filenames,
            "trend_analysis": {
                "slope": trend_slope,
                "interpretation": f"The data shows a {trend_interpretation} with an average monthly increase of {trend_slope:.2f}."
            }
        }
    )
    except Exception as e:
        return json.dumps({"status": "Failure", "message": f"An error occurred: {e}"})

# === TOOL 3: Stationarity Analysis ===
def run_adf_test(timeseries_json: str) -> str:
    """
    TOOL 4: Performs the Augmented Dickey-Fuller (ADF) test to check for stationarity.
    
    Args:
        timeseries_json (str): JSON string of the time series to test.

    Returns:
        str: A JSON string with the ADF test p-value and interpretation.
    """
    print("AGENT ACTION: Executing 'run_adf_test'...")
    s = pd.read_json(timeseries_json, typ='series')
    result = adfuller(s)
    p_value = result[1]
    is_stationary = p_value <= 0.05
    interpretation = "The series is likely stationary (p <= 0.05)." if is_stationary else "The series is likely non-stationary (p > 0.05)."

    return json.dumps({
        "p_value": float(p_value),
        "is_stationary": bool(is_stationary),
        "interpretation": interpretation
    })
def apply_differencing(timeseries_json: str) -> str:
    print("AGENT ACTION: Executing 'apply_differencing'...")
    s = pd.read_json(timeseries_json, typ='series')
    differenced_series = s.diff().dropna()
    return json.dumps({"differenced_timeseries": differenced_series.to_json(date_format='iso')})


def run_decomposition(temp_filepath: str, seasonal_period: int, plot_dir: str) -> str:
    """
    TOOL: Intelligently performs seasonal decomposition. It validates the data and automatically
    chooses the best model (additive or multiplicative), saves the plot, and returns the choice.
    """
    print("AGENT ACTION: Executing 'run_decomposition'...")
    try:
        s = pd.read_json(temp_filepath, typ='series')
        os.makedirs(plot_dir, exist_ok=True)
        _set_plot_style()

        if (s <= 0).any():
            print("...Data contains non-positive values. Defaulting to additive decomposition.")
            chosen_model = 'additive'
            winner_decomp = seasonal_decompose(s, model='additive', period=seasonal_period)
            reason = "Multiplicative model is not appropriate for datasets with zero or negative values. Defaulted to additive."
        else:
            print("...Data is all positive. Comparing additive and multiplicative models.")
            additive_decomp = seasonal_decompose(s, model='additive', period=seasonal_period)
            multiplicative_decomp = seasonal_decompose(s, model='multiplicative', period=seasonal_period)

            add_resid_var = additive_decomp.resid.var(ddof=0)
            mul_resid_var = multiplicative_decomp.resid.var(ddof=0)

            if add_resid_var < mul_resid_var:
                chosen_model = 'additive'
                winner_decomp = additive_decomp
                reason = f"Additive model chosen (resid var: {add_resid_var:.2f}) over Multiplicative (resid var: {mul_resid_var:.2f})."
            else:
                chosen_model = 'multiplicative'
                winner_decomp = multiplicative_decomp
                reason = f"Multiplicative model chosen (resid var: {mul_resid_var:.2f}) over Additive (resid var: {add_resid_var:.2f})."
        
        fig = winner_decomp.plot(); fig.set_size_inches(14, 10);
        fig.suptitle(f'Seasonal Decomposition ({chosen_model.capitalize()})', fontsize=18, y=0.95)
        filename = "05_decomposition_plot.png" 
        plt.savefig(os.path.join(plot_dir, filename)); plt.close()

        return json.dumps({
            "status": "Success",
            "chosen_model": chosen_model,
            "reason": reason,
            "plot_filename": filename
        })
    except Exception as e:
        return json.dumps({"status": "Failure", "message": f"An error occurred during decomposition: {e}"})


# === TOOL 4: Model Identification ===
def analyze_acf_pacf(timeseries_json: str, plot_dir: str) -> str:
    print("AGENT ACTION: Executing 'analyze_acf_pacf'...")
    s = pd.read_json(timeseries_json, typ='series')
    os.makedirs(plot_dir, exist_ok=True)
    from statsmodels.tsa.stattools import acf, pacf
    acf_values, acf_confint = acf(s, nlags=20, alpha=0.05)
    pacf_values, pacf_confint = pacf(s, nlags=20, alpha=0.05)
    acf_sig_lags = [i for i, val in enumerate(acf_values) if i > 0 and abs(val) > (acf_confint[i][1] - val)]
    pacf_sig_lags = [i for i, val in enumerate(pacf_values) if i > 0 and abs(val) > (pacf_confint[i][1] - val)]
    _set_plot_style()
    fig, axes = plt.subplots(2, 1); plot_acf(s, ax=axes[0], lags=20); plot_pacf(s, ax=axes[1], lags=20)
    plt.savefig(os.path.join(plot_dir, "05_acf_pacf_plots.png")); plt.close()
    return json.dumps({"status": "Success", "acf_significant_lags": acf_sig_lags, "pacf_significant_lags": pacf_sig_lags})

# === TOOL 5: Modeling & Validation ===

def find_best_sarima_model_with_grid_search(timeseries_json: str, seasonal_period: int, model_dir: str, model_filename: str) -> str:
    """
    TOOL: Performs a grid search to find the best SARIMA model with the lowest AIC.
    It searches a predefined space of p, q, P, and Q parameters.
    Saves the best model found to a file.
    
    Args:
        timeseries_json (str): JSON of the ORIGINAL (non-differenced) time series.
        seasonal_period (int): The seasonal period (e.g., 12 for monthly).
        model_dir (str): Directory to save the final model.
        model_filename (str): Filename for the saved model.

    Returns:
        str: A JSON object with the best model's order, AIC, and saved file path.
    """
    print(f"AGENT ACTION: Executing 'find_best_sarima_model_with_grid_search'...")
    s = pd.read_json(timeseries_json, typ='series')
    os.makedirs(model_dir, exist_ok=True)
    
    p = d = q = range(0, 2) # p, q
    pdq = list(itertools.product(p, [1], q)) 
    seasonal_pdq = [(x[0], x[1], x[2], seasonal_period) for x in list(itertools.product(p, [1], q))] 

    best_aic, best_order, best_seasonal_order, best_model = float("inf"), None, None, None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(s, order=param, seasonal_order=param_seasonal).fit(disp=False)
                if model.aic < best_aic:
                    best_aic, best_order, best_seasonal_order, best_model = model.aic, param, param_seasonal, model
            except Exception:
                continue
            
    if best_model is None:
        return json.dumps({"status": "Failure", "message": "Grid search failed to find a suitable SARIMA model."})

    model_filepath = os.path.join(model_dir, model_filename)
    with open(model_filepath, 'wb') as pkl_file:
        pickle.dump(best_model, pkl_file)

    return json.dumps({
        "status": "Success",
        "best_model_order": {"order": list(best_order), "seasonal_order": list(best_seasonal_order)},
        "best_model_aic": best_aic,
        "saved_model_path": model_filepath
    })

def run_model_diagnostics(model_filepath: str) -> str:
    print("AGENT ACTION: Executing 'run_model_diagnostics'...")
    with open(model_filepath, 'rb') as f:
        model_results = pickle.load(f)
    diagnostics = model_results.summary().tables[2].data
    ljung_box_prob_q = float(diagnostics[1][3]); jarque_bera_prob_jb = float(diagnostics[2][3])
    lb_passed = ljung_box_prob_q > 0.05; jb_passed = jarque_bera_prob_jb > 0.05
    interpretation = "Model is well-fitted." if lb_passed and jb_passed else "Model may not be well-fitted."
    return json.dumps({"ljung_box_p_value": ljung_box_prob_q, "jarque_bera_p_value": jarque_bera_prob_jb, "residuals_are_uncorrelated": lb_passed, "residuals_are_normal": jb_passed, "interpretation": interpretation})

# === TOOL 6: Forecasting & Reporting Prep ===
def generate_forecast(model_filepath: str, forecast_steps: int, plot_dir: str) -> str:
    """
    TOOL 10: Loads a fitted model and generates a forecast for a specified number of steps.
    Saves a plot of the forecast.
    
    Args:
        model_filepath (str): Path to the saved .pkl model file.
        forecast_steps (int): Number of future periods to forecast.
        plot_dir (str): Directory to save the forecast plot.

    Returns:
        str: A JSON object with the forecast values and confidence intervals.
    """
    print(f"AGENT ACTION: Executing 'generate_forecast' for {forecast_steps} steps...")
    os.makedirs(plot_dir, exist_ok=True)
    with open(model_filepath, 'rb') as f:
        model: SARIMAXResults = pickle.load(f)
        
    historical_values = model.model.endog
    historical_index = model.fittedvalues.index

    forecast_obj = model.get_forecast(steps=forecast_steps)
    forecast_values = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    last_historical_date = historical_index.max()
    forecast_index = pd.date_range(
        start=last_historical_date + pd.DateOffset(months=1),
        periods=forecast_steps,
        freq='MS' 
    )
    
    forecast_values.index = forecast_index
    conf_int.index = forecast_index

    _set_plot_style()
    plt.figure()
    
    plt.plot(historical_index, historical_values, label='Historical Sales')
    plt.plot(forecast_values.index, forecast_values.values, label='Forecasted Sales', color='red', linestyle='--')

    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.7)
    plt.title(f'{forecast_steps}-Month Sales Forecast'); plt.legend()
    plt.savefig(os.path.join(plot_dir, "07_forecast_plot.png")); plt.close()

    forecast_dict = {date.isoformat(): value for date, value in forecast_values.items()}
    conf_int_dict = {
        'lower_y': {date.isoformat(): value for date, value in conf_int['lower y'].items()},
        'upper_y': {date.isoformat(): value for date, value in conf_int['upper y'].items()}
    }

    return json.dumps({
        "status": "Success",
        "message": f"Forecast generated and plot saved.",
        "forecast_values": forecast_dict,
        "confidence_intervals": conf_int_dict
    })
    
def compile_final_report_context(full_context: dict, log_filepath: str) -> str:
    print(f"AGENT ACTION: Executing 'compile_final_report_context' and saving to '{log_filepath}'...")
    try:
        with open(log_filepath, 'w') as f:
            json.dump(full_context, f, indent=4)
        return json.dumps(full_context, indent=4)
    except Exception as e:
        return json.dumps({"status": "Failure", "message": f"Error compiling or saving report context: {e}"})

# === TOOL 7: Final Report Generation ===
def _get_llm():
    if not hasattr(_get_llm, "llm"):
        api_key = os.getenv("GOOGLE_API_KEY")
        _get_llm.llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.5)
    return _get_llm.llm

def generate_summary_report(full_context_json: str, audience_type: str, output_filepath: str) -> str:
    """
    TOOL 12: Generates a final summary report for a specific audience (technical or non-technical).
    Use this AFTER you have compiled the final context.
    
    Args:
        full_context_json (str): The full analysis context from the 'compile_final_report_context' tool.
        audience_type (str): Must be either 'technical' or 'non-technical'.
        output_filepath (str): The path where the final report should be saved.

    Returns:
        str: A confirmation message that the report was successfully generated and saved.
    """
    print(f"AGENT ACTION: Executing 'generate_summary_report' for '{audience_type}' audience...")
    
    technical_template = """
    You are an expert AI Data Scientist. Your task is to provide a concise, technical summary of a time series analysis.
    Based on the following context, write the report.
    
    CONTEXT: {context}
    
    **Instructions:**
    1.  **Objective:** State the goal of the analysis.
    2.  **Methodology:** Detail the key steps (stationarity, differencing, model selection).
    3.  **Model Details:** State the final SARIMA model order and its AIC score.
    4.  **Conclusion:** Conclude with the model's reliability based on diagnostic checks.
    """
    
    non_technical_template = """
    You are a senior business analyst. Your task is to present a clear, actionable summary to management. Avoid all technical jargon.
    Based on the following context, write the report.
    
    CONTEXT: {context}
    
    **Instructions:**
    1.  **Executive Summary:** Start with the key business takeaway: Are sales growing and what is the outlook?
    2.  **Trends and Patterns:** In simple terms, describe the overall sales trend and any seasonal patterns.
    3.  **Sales Projections:** Describe the forecast in business terms and provide actionable advice.
    """
    
    try:
        llm = _get_llm(); context_dict = json.loads(full_context_json)
        prompt = technical_template if audience_type == 'technical' else non_technical_template
        report_text = llm.invoke(prompt.format(context=json.dumps(context_dict, indent=2)))
        with open(output_filepath, 'w') as f:
            f.write(report_text)
        return f"Successfully generated and saved the {audience_type} report to '{output_filepath}'."
    except Exception as e:
        return f"Failed to generate report: {e}"