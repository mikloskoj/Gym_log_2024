import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


# Get the directory where the Python script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Path to the CSV file (relative to the script directory)
csv_file_name = 'gym_log_Q1_2024 - bio_data.csv'
file_path = os.path.join(script_directory, csv_file_name)

def load_and_process_gym_log(filepath: str) -> pd.DataFrame:
   
    try:
        # Read the CSV file
        bio_data_df = pd.read_csv(file_path, sep=';', index_col=0)
        bio_data_df = bio_data_df.map(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
        
        # Convert columns to numeric, coercing errors
        bio_data_df['Wgt (kg)'] = pd.to_numeric(bio_data_df['Wgt (kg)'], errors='coerce')
        bio_data_df['Waist (cm)'] = pd.to_numeric(bio_data_df['Waist (cm)'], errors='coerce')
        bio_data_df['kcal Total'] = pd.to_numeric(bio_data_df['kcal Total'], errors='coerce')
        bio_data_df['kcal'] = pd.to_numeric(bio_data_df['kcal'], errors='coerce')
        
        # Convert index to datetime for better handling of date ticks
        bio_data_df.index = pd.to_datetime(bio_data_df.index, dayfirst=True)


    except FileNotFoundError:
        print(f"Error: The file {csv_file_name} was not found in the directory {script_directory}.")
    except pd.errors.ParserError:
        print("Error: There was an issue parsing the CSV file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return bio_data_df


def data_transformation(df) -> None:
    print(f'Transforming the data...')


def moving_average(series, window_size):
    '''Calculating moving_average'''
    return series.rolling(window=window_size, min_periods=1).mean()


def bio_plot(df) -> None:
    try:
        # Create a figure and subplots
        fig, (plt1, plt2) = plt.subplots(2, 1, figsize=(10, 8))

        # Apply moving average to smooth the data
        weight_smoothed = moving_average(df['Wgt (kg)'], window_size=5)
        waist_smoothed = moving_average(df['Waist (cm)'], window_size=5)

        # Plot weight and waist measurements on the first subplot
        plt1.plot(df.index, weight_smoothed, label='Weight (kg)', color='#0b5193')
        plt1.plot(df.index, waist_smoothed, label='Waist (cm)', color='#930b0b')
        plt1.grid()
        plt1.legend(loc='upper left')
        plt1.set_xticks(df.index[::7])
        plt1.tick_params(axis='x', rotation=45, labelsize=7)
        plt1.tick_params(axis='y', labelsize=7)
        plt1.set_title('Weight and Waist Measurements Over Time  (moving_average)')

        # Highlight the dates with 'Creatine' == 1 on the first subplot
        creatine_dates = df[df['Creatine'] == 1].index
        for date in creatine_dates:
            plt1.axvline(x=date, color='lightgrey', linestyle='--', alpha=0.5)

        # Add a proxy artist for the creatine marker to the legend
        creatine_proxy = plt.Line2D([0], [0], color='lightgrey', linestyle='--', alpha=0.5, label='Creatine')
        plt1.legend(handles=[plt.Line2D([], [], color='#0b5193', label='Weight (kg)'),
                            plt.Line2D([], [], color='#930b0b', label='Waist (cm)'),
                            creatine_proxy], loc='lower left', fontsize=8)

        # Adding horizontal lines at 1800 and 2000
        plt2.axhline(y=1750, color='#9a1d04', linestyle=':', linewidth=1.5, label='Basal metabolic rate - Caloric target')
        plt2.axhline(y=1950, color='#9a1d04', linestyle=':', linewidth=1.5)

        # Plot kcal Total and kcal as bar charts on the second subplot
        plt2.bar(df.index, df['kcal Total'], label='Total kcal consumed', alpha=0.7, color='#f2ba02')
        plt2.bar(df.index, df['kcal'], label='Total kcal consumed (After excercise)', alpha=0.7, color='#2f6624')
        plt2.grid()
        plt2.legend(loc='upper left', fontsize=8)
        plt2.set_xticks(df.index[::7])
        plt2.tick_params(axis='x', rotation=45, labelsize=5)
        plt2.tick_params(axis='y', labelsize=5)
        plt2.set_title('Caloric Intake Over Time')

        # Adjust the layout to make room for rotated labels
        # fig.tight_layout(rect=[0, 0.8, 1, 0.95])  # Adjust layout to make room for the description
        fig.subplots_adjust(hspace=1.0)

        description2 = (
            "The second plot shows the caloric intake over the first quarter of 2024.\n"
            "Total caloric intake (kcal Total) and caloric intake (kcal) are displayed as bar charts.\n"
            "Missing values have been handled appropriately to ensure continuous data representation."
        )
        fig.text(0.1, 0.45, description2, wrap=True, horizontalalignment='left', fontsize=7, color='grey')

        print('Displaying the plot...')
        plt.show()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")



def main() -> None:
    # Pass the cleaned DataFrame to the function
    load_and_process_gym_log(file_path)
    bio_plot(pd.DataFrame)

main()