import pandas as pd
import matplotlib.pyplot as plt
import os

def main_statistics():
    # Get the directory where the Python script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Path to the CSV file (relative to the script directory)
    csv_file_name = 'gym_log_Q1_2024 - bio_data.csv'
    file_path = os.path.join(script_directory, csv_file_name)

    try:

        # Read the CSV file
        df = pd.read_csv(file_path, sep=';', index_col=0)

        print(df.head())
        print("\nBasic Statistics:")
        print(df.describe())


    except FileNotFoundError:
        print(f"Error: The file {csv_file_name} was not found in the directory {script_directory}.")
    except pd.errors.ParserError:
        print("Error: There was an issue parsing the CSV file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

main_statistics()