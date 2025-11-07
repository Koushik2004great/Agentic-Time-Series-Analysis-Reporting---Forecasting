import os
import json
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent

from src.scribe.tools import *

def setup_llm():
    """Initializes the Gemini LLM."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("ERROR: GOOGLE_API_KEY not found in .env file.")
    try:
        model = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.0)
        print("Gemini 2.0 Flash model initialized successfully.")
        return model
    except Exception as e:
        raise ConnectionError(f"ERROR: Failed to initialize the Gemini model: {e}")

AGENT_PROMPT_TEMPLATE = """
You are Scribe, an expert and interactive AI Data Scientist. Your goal is to guide a user through analyzing a CSV file for time series forecasting.

**METHODOLOGY:**
1.  **List & Select File:** First, find the available CSV files and ask the user to choose one.
2.  **Get Columns:** Next, get the column details for the chosen file.
3.  **Identify Columns:** Based on the details, make an educated guess for the `date_column` and `value_column`.
4.  **Find & Ask Filter:** Use the `Find and Ask Categorical Filter` tool. This will automatically find a filterable column (if one exists) and ask the user for their choice.
5.  **Load Data:** Use the `Load and Prepare Data` tool, providing all the parameters you have gathered.
6.  **Execute Full Analysis:** Proceed with the full time series analysis workflow.

**TOOLS:**
{tools}

**AVAILABLE COMMANDS:**
{tool_names}

**OUTPUT FORMAT AND EXAMPLE:**
You MUST use the following format. Pay close attention to the `Action Input` format.

```
Thought: I need to ask the user to choose from a list of files. I will format the input for the 'Ask User for Choice' tool as a valid JSON string.
Action: Ask User for Choice
Action Input: {{"question": "Which CSV file would you like to analyze?", "options": ["file1.csv", "file2.csv"]}}
Observation: The user chose to proceed with 'file1.csv'.
```

**CRITICAL RULE:** The `Action Input` for any tool that requires a JSON string MUST be a valid, single-line JSON object, just like in the example above. For other tools, provide the input as a simple string.

**FINAL ANSWER FORMAT:**
When you have successfully generated BOTH reports, your job is complete. You must use the following format:
```
Thought: I have now generated and saved both the technical and non-technical reports. My work is complete.
Final Answer: Analysis and reporting complete. All artifacts saved.
```

Begin!

**Current Task:**
{input}

**Chain of Thought History:**
{agent_scratchpad}
"""

def run_ai_analyst(config: dict):
    """
    Initializes and runs the autonomous, interactive AI Data Scientist agent.
    """
    print("--- Initializing Project Scribe v3.0 Interactive Analyst ---")
    try:
        llm = setup_llm()
        
        paths = config['paths']; params = config['parameters']; filenames = config['filenames']
        input_dir = paths['input_directory']
        plot_dir = os.path.join(paths['output_directory'], paths['plot_subdirectory'])
        model_dir = os.path.join(paths['output_directory'], paths['model_directory'])
        report_dir = os.path.join(paths['output_directory'], paths['report_subdirectory'])
        log_filepath = os.path.join(paths['output_directory'], paths['log_subdirectory'], filenames['final_log'])
        temp_dir = os.path.join(paths['output_directory'], 'temp')

        def _get_column_details_wrapper(filename: str) -> str:
            """Wrapper to construct the full path for getting column details."""
            full_path = os.path.join(input_dir, filename)
            return get_csv_column_details(full_path)

        def _find_and_ask_wrapper(filename: str) -> str:
            """Wrapper to construct the full path for finding categories."""
            full_path = os.path.join(input_dir, filename)
            return find_and_ask_categorical_filter(full_path)

        def _load_data_wrapper(params_str: str) -> str:
            """Wrapper that takes the agent's parameters and adds the full filepath."""
            params_dict = json.loads(params_str)
            filename = params_dict.get('filename')
            if not filename:
                return json.dumps({"status": "Failure", "message": "The 'filename' key is missing in the input."})
            
            full_path = os.path.join(input_dir, filename)
            final_params = {**params_dict, 'filepath': full_path}
            
            return load_and_prepare_data(final_params, temp_dir)

        tools = [
            Tool(name="List CSV Files", func=lambda x: list_csv_files_in_directory(input_dir), description="Use this first to find all available CSV files in the input directory."),
            Tool(name="Ask User for Choice", func=ask_user_for_choice, description="Asks the user to choose from a list of options. Input must be a JSON string with keys 'question' and 'options'."),
            
            Tool(name="Get CSV Column Details", func=_get_column_details_wrapper, description="Inspects a specific CSV file. The input must be only the FILENAME chosen by the user."),
            Tool(name="Find and Ask Categorical Filter", func=_find_and_ask_wrapper, description="Automatically finds a filterable category and asks the user for a choice. The input must be only the FILENAME chosen by the user."),
            
            Tool(name="Load and Prepare Data", func=_load_data_wrapper, description="""Loads and prepares the time series. The input MUST be a JSON string with these EXACT keys: 'filename', 'date_col', 'value_col'. Optional keys are 'category_col' and 'category_to_filter'."""),
            
            Tool(name="Perform EDA and Analyze Trend", func=lambda filepath: perform_eda_and_save_plots(filepath, plot_dir), description="Performs EDA. Input must be the `temp_timeseries_path`."),
            Tool(name="Run ADF Test for Stationarity", func=run_adf_test, description="Performs an ADF test. Input must be the `temp_timeseries_path`."),
            Tool(name="Run Intelligent Decomposition", func=lambda filepath: run_decomposition(filepath, params['seasonal_period'], plot_dir), description="Automatically determines the best decomposition model. Input must be the `temp_timeseries_path`."),
            Tool(name="Analyze ACF and PACF", func=lambda filepath: analyze_acf_pacf(filepath, plot_dir), description="Calculates significant lags from ACF/PACF plots. Input must be the `temp_timeseries_path`."),
            Tool(name="Find Best SARIMA Model with Grid Search", func=lambda filepath: find_best_sarima_model_with_grid_search(filepath, params['seasonal_period'], model_dir, filenames['final_model']), description="Performs an automated grid search to find the best SARIMA model. Input must be the `temp_timeseries_path`."),
            Tool(name="Run Model Diagnostics", func=lambda x: run_model_diagnostics(os.path.join(model_dir, filenames['final_model'])), description="Checks a fitted model's residuals."),
            Tool(name="Generate Forecast", func=lambda x: generate_forecast(os.path.join(model_dir, filenames['final_model']), params['forecast_steps'], plot_dir), description="Generates a forecast using the validated model."),
            Tool(name="Compile Final Report Context", func=lambda context_dict: compile_final_report_context(context_dict, log_filepath), description="Gathers all findings and saves the final analysis log."),
            Tool(name="Generate Technical Summary Report", func=lambda context_json: generate_summary_report(context_json, 'technical', os.path.join(report_dir, filenames['technical_report'])), description="Generates the final report for a technical audience."),
            Tool(name="Generate Non-Technical Summary Report", func=lambda context_json: generate_summary_report(context_json, 'non-technical', os.path.join(report_dir, filenames['non_technical_report'])), description="Generates the final report for a non-technical, business audience.")
        ]

        prompt = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE)
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=30)
        
        initial_goal = "Guide the user through selecting and analyzing a time series dataset from the available files."
        print(f"\n--- AI Analyst Starting Workflow --- \nGoal: {initial_goal}\n")
        
        result = agent_executor.invoke({"input": initial_goal})
        
        print("\n--- AI Analyst Workflow Complete ---")
        print(f"Final Result: {result['output']}")

    except Exception as e:
        print(f"\n--- An unexpected error occurred. --- \nError Details: {e}")