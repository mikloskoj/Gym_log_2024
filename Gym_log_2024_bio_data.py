import pandas as pd
import matplotlib.pyplot as plt
import os

def bio_data():

    # Get the directory where the Python script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Path to the CSV file (relative to the script directory)
    csv_file_name = 'gym_log_Q1_2024 - bio_data.csv'
    file_path = os.path.join(script_directory, csv_file_name)

    # Read the CSV file
    df = pd.read_csv(file_path, sep=';', index_col=0)

    # Convert columns to numeric, coercing errors
    df['Wgt (kg)'] = pd.to_numeric(df['Wgt (kg)'], errors='coerce')
    df['Waist (cm)'] = pd.to_numeric(df['Waist (cm)'], errors='coerce')
    df['kcal Total'] = pd.to_numeric(df['kcal Total'], errors='coerce')
    df['kcal'] = pd.to_numeric(df['kcal'], errors='coerce')

    # Convert index to datetime for better handling of date ticks
    df.index = pd.to_datetime(df.index, dayfirst=True)

    # Create a figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Plot weight and waist measurements on the first subplot
    ax1.plot(df['Wgt (kg)'], label='Weight (kg)')
    ax1.plot(df['Waist (cm)'], label='Waist (cm)')
    ax1.grid()
    ax1.legend(loc='upper left')
    ax1.set_xticks(df.index[::7])
    ax1.tick_params(axis='x', rotation=45, labelsize=5)
    ax1.tick_params(axis='y', labelsize=5)
    ax1.set_title('Weight and Waist Measurements Over Time')

    # Highlight the dates with 'Creatine' == 1 on the first subplot
    creatine_dates = df[df['Creatine'] == 1].index
    for date in creatine_dates:
        ax1.axvline(x=date, color='lightgrey', linestyle='--', alpha=0.5)

    # Add a proxy artist for the creatine marker to the legend
    creatine_proxy = plt.Line2D([0], [0], color='lightgrey', linestyle='--', alpha=0.5, label='Creatine')
    ax1.legend(handles=[plt.Line2D([], [], color='blue', label='Weight (kg)'),
                        plt.Line2D([], [], color='orange', label='Waist (cm)'),
                        creatine_proxy], loc='lower left', fontsize=8)

    # Adding horizontal lines at 1800 and 2000
    ax2.axhline(y=1800, color='darkgreen', linestyle='--', linewidth=1, label='1800 kcal')
    ax2.axhline(y=2000, color='darkgreen', linestyle='--', linewidth=1, label='2000 kcal')

    # Plot kcal Total and kcal as bar charts on the second subplot
    ax2.bar(df.index, df['kcal Total'], label='kcal Total', alpha=0.7)
    ax2.bar(df.index, df['kcal'], label='kcal', alpha=0.7)
    ax2.grid()
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_xticks(df.index[::7])
    ax2.tick_params(axis='x', rotation=45, labelsize=5)
    ax2.tick_params(axis='y', labelsize=5)
    ax2.set_title('Caloric Intake Over Time')

    # Adjust the layout to make room for rotated labels
    fig.tight_layout(rect=[0, 0.8, 1, 0.95])  # Adjust layout to make room for the description
    fig.subplots_adjust(hspace=1.0)

    description2 = (
        "The second plot shows the caloric intake over the first quarter of 2024.\n"
        "Total caloric intake (kcal Total) and caloric intake (kcal) are displayed as bar charts.\n"
        "Missing values have been handled appropriately to ensure continuous data representation."
    )
    fig.text(0.1, 0.45, description2, wrap=True, horizontalalignment='left', fontsize=7, color='grey')

    # Provide options to the user
    print('For linechart press 1, to save plot as PNG press 2')

    q = input('Your choice: ')
    if q == '1':
        print('Displaying the plot...')
        plt.show()
    elif q == '2':
        # Construct the save path
        save_path = os.path.join(script_directory, 'weight_waist_kcal_plot.png')
        fig.savefig(save_path)
        print(f'Plot saved as PNG to {save_path}')
    else:
        print('-------------------------------------------\n'
            'You did not select anything from the menu.\n'
            'End of program... Goodbye and thank you!\n'
            '-------------------------------------------')
        
bio_data()
