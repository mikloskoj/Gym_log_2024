import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r'C:\Users\janmi\Documents\VS Code\Gym_log_2024\gym_log_Q1_2024 - workout data.csv'
body_weight = 79

# outliers = ['Plank']
selected_exercises = ('Kneeling dip', 'Bench press', 'Chest press', 'Prone leg curl', 'Lat pulldown', 'Bicep curl')

def filter_remove_duplicities(df) -> None:
    filtered_df = df[df['Muscle group'] != "Cardio"]
    filtered_df = filtered_df['Muscle group'].drop_duplicates()
    print(filtered_df)

def total_volumes(ax1, ax2, df, body_weight, selected_exercises, window=8) -> None:
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    
    # Filter the dataframe based on muscle group and selected exercises
    # filtered_df = df[(df['Muscle group'] != 'Cardio') & (df['Muscle group'] != 'Walk')].copy()
    filtered_df = df[(df['Muscle group'] != 'Cardio') & (df['Muscle group'] != 'Walk') & (df['Exercise name'].isin(selected_exercises))].copy()
    
    # Replace commas in 'Weight' column and convert to numeric
    filtered_df['Weight'] = filtered_df['Weight'].str.replace(',', '.')
    filtered_df['Weight'] = pd.to_numeric(filtered_df['Weight'], errors='coerce')
    filtered_df['Reps'] = pd.to_numeric(filtered_df['Reps'], errors='coerce')
    filtered_df['Sets'] = pd.to_numeric(filtered_df['Sets'], errors='coerce')
    
    # Adjust weights for body weight exercises
    filtered_df.loc[filtered_df['Body weight flg'] == 'BW', 'Weight'] += body_weight
    
    # Calculate total reps and total weight
    filtered_df['total_reps'] = filtered_df['Reps'] * filtered_df['Sets']
    filtered_df['total_weight'] = filtered_df['Reps'] * filtered_df['Sets'] * filtered_df['Weight']
    
    # Group by 'Exercise name' and sum the relevant columns for total summary
    grouped_df_total = filtered_df.groupby(['Exercise name'])[['Sets', 'Reps', 'Weight', 'total_reps', 'total_weight']].sum()
    grouped_df_total = grouped_df_total.sort_values('total_reps', ascending=False)

    # Keep only the top 3 exercises
    top_exercises = grouped_df_total.head(5)

    # Filter the main dataframe to keep only the top 3 exercises
    filtered_df = filtered_df[filtered_df['Exercise name'].isin(top_exercises.index)]

    # Update grouped_df with only the top 3 exercises
    grouped_df = filtered_df.groupby(['Date', 'Exercise name'])[['Sets', 'Reps', 'Weight', 'total_reps', 'total_weight']].sum().reset_index()

    # Apply moving average
    grouped_df['total_reps_ma'] = grouped_df.groupby('Exercise name')['total_reps'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

    # Create a palette and use it for both plots based on sorted total reps
    unique_exercises = top_exercises.index
    palette = sns.color_palette('YlOrRd_r', len(unique_exercises))
    color_mapping = dict(zip(unique_exercises, palette))

    # Plot line plot with moving average
    sns.lineplot(
        ax=ax1,
        data=grouped_df,
        x='Date',
        y='total_reps_ma',
        hue='Exercise name',
        palette=color_mapping, 
        marker='v'
    )

    ax1.tick_params(axis='x', rotation=45)

    # Plot bar plot
    sns.barplot(
        data=top_exercises.reset_index(),
        x='Exercise name',
        y='total_reps',
        ax=ax2,
        palette=color_mapping
    )

    ax2.set_xlabel('Exercise')
    ax2.set_ylabel('Total Reps')
    ax2.set_title('Total Reps Lifted Over Time', fontweight='bold')
    
    # Add table
    table_data = top_exercises.reset_index()
    table = ax2.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='bottom', bbox=[0, -0.5, 1, 0.34])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 0.5)
    
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=6)
            cell.set_facecolor('#40466e')
        else:
            cell.set_facecolor('#f2f2f2')
    
    plt.subplots_adjust(left=0.1, bottom=0.2)

def main() -> None:
    try:
        df = pd.read_csv(file_path, sep=';', index_col=None, encoding='latin1')
    except FileNotFoundError as e:
        print(f"File not found. Details: {e}")
        return

    # Gym Workout Analysis 2024 - Total volumes
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(14, 7))
    total_volumes(ax1, ax2, df, body_weight, selected_exercises)
    fig.text(0.95, 0.1, f'Gym Workout Analysis Q1 2024\nThis shows the total volumes I have lifted since 1st of January.\nI hope you like my charts and everything is clear.\nselected_exercises: {selected_exercises} were excluded.', ha='right', va='center', fontsize=10, fontweight='normal')
    fig.canvas.manager.set_window_title('Gym Workout Analysis 2024 - Total volumes')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
