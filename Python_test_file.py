import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


df = pd.read_excel(r'C:\Users\janmi\Documents\VS Code\Gym_log_2024\gym_log_Q1_2024 - workout data.xlsx')
df = df.sort_values('Date', ascending=True)


body_weight = 79


# If the weight is negative then calculate body weight - weight
# Negative weight is when the exercise is supported.
df['Weight'] = df['Weight'].apply(lambda x: body_weight + x if x < 0 else x)

# If the exercise is with body weight then calculate body weight + weight
df.loc[df['Body weight flg'] == 'BW', 'Weight'] = df.loc[df['Body weight flg'] == 'BW', 'Weight'].apply(lambda x: body_weight + x)

# Exercises to compare
exercises_to_compare = ["Kneeling dip", "Squat", "Bench press"]


def plot_weight_over_time(df, exercises):
    """Plot the weight over time for selected exercises using smoothed line charts and display a summary table."""
    
    fig, (plt1, plt2, plt3) = plt.subplots(3, 1, figsize=(12, 12))

    # Define colors for specific exercises
    color_map = {
        'Kneeling dip': '#2f6624',
        'Squat': '#f2ba02',  # You can add more exercises and their colors here
        # Add other exercises and their colors if needed
    }

    for exercise_name in exercises:
        filtered_df = df[df['Exercise name'] == exercise_name]
        
        # Group by Date and find the max weight for each date
        max_weights = filtered_df.groupby('Date')['Weight'].max().reset_index()

        # Apply Savitzky-Golay filter to smooth the lines
        smoothed_weights = savgol_filter(max_weights['Weight'], window_length=5, polyorder=2)

        # Get color for the exercise, default to None if not in the map
        color = color_map.get(exercise_name, None)
        
        # Plot the smoothed line chart on the first subplot
        plt1.plot(max_weights['Date'], smoothed_weights, label=exercise_name, linewidth=3, alpha=0.7, color=color)

    plt1.set_title('Weight Over Time for Selected Exercises')
    plt1.set_xlabel('Date')
    plt1.set_ylabel('Weight (Kg)')
    plt1.legend()
    plt1.grid(True)

    # Placeholder for the second subplot (you can customize it as needed)
    plt2.set_title('Placeholder for Second Plot')
    plt2.grid(True)

    # Create summary table data
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

    # Plot the summary table on the third subplot
    plt3.axis('tight')
    plt3.axis('off')
    table = plt3.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='center', loc='center')

    # Make the header bold
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header cells
            cell.set_text_props(weight='bold', color='Grey', fontsize=7)
        
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)  # Set the line width of the cell borders

    plt3.set_title('Summary Table of Max and Min Weights for Each Exercise')
    
    fig.subplots_adjust(hspace=0.5)
    plt.show()


# Call the function to display the plots
plot_weight_over_time(df, exercises_to_compare)