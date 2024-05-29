import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the Excel file into a DataFrame
df = pd.read_excel(r'C:\Users\jmiklosk\OneDrive - DPDHL\Desktop\Pracovní DHL\SuperUser\DATA\Gym_log_2024 temporary\gym_log_Q1_2024 - workout data.xlsx')

# Sort the DataFrame by the 'Date' column in ascending order
df = df.sort_values('Date', ascending=True)

# Define the body weight to use for replacing negative values
body_weight = 79

# Replace negative values in the 'Weight' column with the body weight
df['Weight'] = df['Weight'].apply(lambda x: body_weight + x if x < 0 else x)

# List of exercises to compare
exercises_to_compare = ["Kneeling dip", "Squat", "Bench press"]

def plot_weight_over_time(df, exercises):
    """Plot the weight over time for selected exercises."""
    plt.figure(figsize=(12, 8))

    for exercise_name in exercises:
        filtered_df = df[df['Exercise name'] == exercise_name]
        plt.plot(filtered_df['Date'], filtered_df['Weight'], marker='o', label=exercise_name)

    plt.title('Weight Over Time for Selected Exercises')
    plt.xlabel('Date')
    plt.ylabel('Weight (Kg)')
    plt.legend()
    plt.grid(True)
    plt.show()

def create_summary_table(df, exercises):
    """Create and display a summary table of max and min weights for each exercise."""
    summary_data = {
        'Exercise': [],
        'Max Weight': [],
        'Min Weight': []
    }

    for exercise_name in exercises:
        filtered_df = df[df['Exercise name'] == exercise_name]
        max_weight = filtered_df['Weight'].max()
        min_weight = filtered_df['Weight'].min()
        summary_data['Exercise'].append(exercise_name)
        summary_data['Max Weight'].append(max_weight)
        summary_data['Min Weight'].append(min_weight)

    summary_df = pd.DataFrame(summary_data)

    # Print the summary table to the console
    print(summary_df)

    # Plot the summary table using matplotlib
    fig, ax = plt.subplots(figsize=(6, 3))  # Set the size of the table plot
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='center', loc='center')

    plt.title('Summary Table of Max and Min Weights for Each Exercise')
    plt.show()

# Call the functions to display the plots
plot_weight_over_time(df, exercises_to_compare)
create_summary_table(df, exercises_to_compare)
