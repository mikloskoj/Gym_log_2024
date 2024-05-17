import pandas as pd
import matplotlib.pyplot as plt
import os

#file_path = r'C:\Users\janmi\Documents\VS Code\Gym_log_2024\gym_log_Q1_2024 - bio_data.csv'
current_directory = os.path.dirname(os.path.abspath(__file__))
csv_file_name = 'gym_log_Q1_2024 - bio_data.csv'
file_path = os.path.join(current_directory, csv_file_name)

df = pd.read_csv(file_path, sep=';', index_col=0)

# Convert 'Wgt (kg)' and 'Waist (cm)' columns to numeric, coercing errors
df['Wgt (kg)'] = pd.to_numeric(df['Wgt (kg)'], errors='coerce')
df['Waist (cm)'] = pd.to_numeric(df['Waist (cm)'], errors='coerce')

# Plot weight and waist measurements
plt.plot(df['Wgt (kg)'], label='Weight (kg)')
plt.plot(df['Waist (cm)'], label='Waist (cm)')
plt.xlabel('Date')
plt.ylabel('Values')
plt.grid()
plt.legend()
plt.xticks(rotation=45, fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()

title = 'Weight and Waist Measurements Over Time'
plt.figtext(0.5, 0.7, title, wrap=True, horizontalalignment='center', fontsize=12, fontweight='bold')

# Select a subset of dates to display on the x-axis ticks
plt.gca().set_xticks(df.index[::7])

# Adjust the position of the plot to lower the top
plt.subplots_adjust(top=0.53)

# Add a description below the plot
description = (
    "This plot shows the changes in weight and waist measurements over the first quarter of 2024.\n"
    "Weight is measured in kilograms (kg) and waist circumference is measured in centimeters (cm).\n"
    "Missing values have been handled appropriately to ensure continuous data representation."
)
plt.figtext(0.5, 0.6, description, wrap=True, horizontalalignment='center', fontsize=8)

print('For linechart press 1, to save plot as PNG press 2')

q = input('Your choice: ')
if q == '1':
    plt.show()
elif q == '2':
    # Get the current working directory
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Construct the save path
    save_path = os.path.join(current_directory, 'weight_waist_plot.png')
    plt.savefig(save_path)
    print(f'Plot saved as PNG to {save_path}')
else:
    print('-------------------------------------------\n'
          'You did not select anything from the menu.\n'
          'End of program... Goodbye and thank you!\n'
          '-------------------------------------------')
