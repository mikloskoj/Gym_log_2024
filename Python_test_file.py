import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

df = pd.read_excel(r'C:\Users\janmi\Documents\VS Code\Gym_log_2024\gym_log_Q1_2024 - workout data.xlsx')
df = df.sort_values('Date', ascending=True)

body_weight = 79

# If the weight is negative then calculate body weight - weight
# Negative weight is when the excercise is supported.
df['Weight'] = df['Weight'].apply(lambda x: body_weight + x if x < 0 else x)

# If the exercise is with body wight then calculate bodyweight + weight
df.loc[df['Body weight flg'] == 'BW', 'Weight'] = df.loc[df['Body weight flg'] == 'BW', 'Weight'].apply(lambda x: body_weight + x)



# Exercises to compare
exercises_to_compare = ["Kneeling dip", "Squat", "Bench press"]


def moving_average(series, window_size):
    '''Calculating moving_average'''
    return series.rolling(window=window_size, min_periods=1).mean()


def plot_weight_over_time(df, exercises):
    """Plot the weight over time for selected exercises."""
    plt.figure(figsize=(12, 8))

    for exercise_name in exercises:
        filtered_df = df[df['Exercise name'] == exercise_name]
        
        # Group by Date and find the max weight for each date
        max_weights = filtered_df.groupby('Date')['Weight'].max().reset_index()

        # Plot the points
        plt.plot(max_weights['Date'], max_weights['Weight'], marker='.', linestyle='-', label=exercise_name)

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
    fig, ax = plt.subplots(figsize=(8, 3))  # Set the size of the table plot
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='center', loc='center')

    # Make the header bold
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header cells
            cell.set_text_props(weight='bold', color='Grey', fontsize=7)
        
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)  # Set the line width of the cell borders

    plt.title('Summary Table of Max and Min Weights for Each Exercise')
    plt.show()

# Call the functions to display the plots
plot_weight_over_time(df, exercises_to_compare)
create_summary_table(df, exercises_to_compare)
