import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

file_path = r'C:\Users\jmiklosk\OneDrive - DPDHL\Desktop\Pracovní DHL\SuperUser\DATA\Gym_log_2024 temporary\gym_log_Q1_2024 - workout data.csv'
body_weight = 79
selected_exercises = ('Kneeling dip', 'Bench press', 'Chest press', 'Prone leg curl', 'Lat pulldown', 'Bicep curl')




def filter_remove_duplicities(df) -> None:
    filtered_df = df[df['Muscle group'] != "Cardio"]
    filtered_df = filtered_df['Muscle group'].drop_duplicates()
    print(filtered_df)

def total_volumes_kneeling_dip(ax, df, body_weight) -> None:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    filtered_df = df[(df['Muscle group'] != 'Cardio') & (df['Muscle group'] != 'Walk')].copy()
    filtered_df = filtered_df[filtered_df['Exercise name'] == 'Kneeling dip']

    filtered_df['Weight'] = filtered_df['Weight'].str.replace(',', '.')
    filtered_df['Weight'] = pd.to_numeric(filtered_df['Weight'], errors='coerce')
    filtered_df['Reps'] = pd.to_numeric(filtered_df['Reps'], errors='coerce')
    filtered_df['Sets'] = pd.to_numeric(filtered_df['Sets'], errors='coerce')

    filtered_df.loc[filtered_df['Body weight flg'] == 'BW', 'Weight'] += body_weight
    filtered_df['Reps_times_Sets'] = filtered_df['Reps'] * filtered_df['Sets']
    filtered_df['total_weight'] = filtered_df['Reps'] * filtered_df['Sets'] * filtered_df['Weight']

    grouped_df = filtered_df.groupby('Date')[['Sets', 'Reps', 'Weight', 'Reps_times_Sets', 'total_weight']].sum()
    
    normalized_weights = (grouped_df['total_weight'] - grouped_df['total_weight'].min()) / (grouped_df['total_weight'].max() - grouped_df['total_weight'].min())
    cmap = plt.get_cmap('YlOrRd')
    colors = cmap(normalized_weights)
    
    ax.bar(grouped_df.index, grouped_df['total_weight'], color=colors)
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Weight')
    ax.set_title('Total Weight Lifted Over Time - Kneeling dip', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)


def total_volumes(ax, df, body_weight, selected_exercises) -> None:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    filtered_df = df[(df['Muscle group'] != 'Cardio') & (df['Muscle group'] != 'Walk') & (df['Exercise name'].isin(selected_exercises))].copy()

    filtered_df['Weight'] = filtered_df['Weight'].str.replace(',', '.')
    filtered_df['Weight'] = pd.to_numeric(filtered_df['Weight'], errors='coerce')
    filtered_df['Reps'] = pd.to_numeric(filtered_df['Reps'], errors='coerce')
    filtered_df['Sets'] = pd.to_numeric(filtered_df['Sets'], errors='coerce')

    filtered_df.loc[filtered_df['Body weight flg'] == 'BW', 'Weight'] += body_weight
    filtered_df['Reps_times_Sets'] = filtered_df['Reps'] * filtered_df['Sets']
    filtered_df['total_weight'] = filtered_df['Reps'] * filtered_df['Sets'] * filtered_df['Weight']

    grouped_df = filtered_df.groupby(['Exercise name'])[['Sets', 'Reps', 'Weight', 'Reps_times_Sets', 'total_weight']].sum()
    grouped_df = grouped_df.sort_values('total_weight', ascending=False)
    
    sns.barplot(data=grouped_df.reset_index(), x='Exercise name', y='total_weight',hue='total_weight' , ax=ax, palette='YlOrRd_r')
    ax.set_xlabel('Exercise')
    ax.set_ylabel('Total Weight')
    ax.set_title('Total Weight Lifted Over Time', fontweight='bold')

    table_data = grouped_df.reset_index()
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='bottom', bbox=[0, -0.5, 1, 0.34])
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
    total_volumes_kneeling_dip(ax1, df, body_weight)
    total_volumes(ax2, df, body_weight, selected_exercises)
    fig.text(0.95, 0.1, 'Gym Workout Analysis Q1 2024\nThis shows the total volumes I have lifted since 1st of January.\nI hope you like my charts and everything is clear.', ha='right', va='center', fontsize=10, fontweight='normal')
    fig.canvas.manager.set_window_title('Gym Workout Analysis 2024 - Total volumes')
    plt.tight_layout()
    plt.show()


    # Different



if __name__ == "__main__":
    main()
