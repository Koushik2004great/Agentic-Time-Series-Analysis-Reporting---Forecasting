import json
import os
from src.scribe.agent import run_ai_analyst

def load_config(config_path='config.json'):
    """Loads the configuration from a JSON file and ensures directories exist."""
    print("--- Loading Project Configuration ---")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Configuration file 'config.json' loaded successfully.")
        
        paths = config.get('paths', {})
        os.makedirs(paths.get('input_directory', 'data/input'), exist_ok=True)
        output_dir = paths.get('output_directory', 'outputs')
        os.makedirs(os.path.join(output_dir, paths.get('log_subdirectory', 'logs')), exist_ok=True)
        os.makedirs(os.path.join(output_dir, paths.get('plot_subdirectory', 'plots')), exist_ok=True)
        os.makedirs(os.path.join(output_dir, paths.get('report_subdirectory', 'reports')), exist_ok=True)
        os.makedirs(os.path.join(output_dir, paths.get('model_directory', 'models')), exist_ok=True)
        print("Ensured all I/O directories exist.")
        
        return config
    except Exception as e:
        print(f"FATAL ERROR loading config: {e}")
        return None

def main():
    """Main execution function for the interactive AI Analyst."""
    print("\n=============================================")
    print("  Initializing Project Scribe v3.0           ")
    print("=============================================")
    
    config = load_config()
    
    if config:
        run_ai_analyst(config)
    else:
        print("\n--- Project execution halted due to configuration errors. ---")
        
    print("\n=============================================")
    print("  Project Scribe v3.0 Execution Finished     ")
    print("=============================================")

if __name__ == '__main__':
    main()