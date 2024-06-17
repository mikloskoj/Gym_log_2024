import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.gridspec as gridspec

file_path_1 = 'gym_log_Q1_2024 - workout data.csv'
file_path_2 = 'gym_log_Q1_2024 - bio_data.csv'
body_weight = 79
height = 181
sns.set_style("white")
border_color = 'lightgrey'
background_color = '#fdfcfc'
color = "#193f71" # color
color_palette = sns.dark_palette(color, reverse=True, as_cmap=True)
line_color = color
selected_exercises = ('Kneeling dip', 'Bench press', 'Chest press', 'Prone leg curl', 'Lat pulldown', 'Bicep curl')
outliers = ['Plank']


def load_data(file_path_1, file_path_2):
    try:
        df1 = pd.read_csv(file_path_1, sep=';', index_col=None, encoding='latin1')
    except FileNotFoundError as e:
        print(f"File not found. Details: {e}")
        return None, None

    try:
        df2 = pd.read_csv(file_path_2, sep=';', index_col=None, encoding='latin1')
    except FileNotFoundError as e:
        print(f"File not found. Details: {e}")
        return df1, None
    
    return df1, df2

def data_preparation(df1, df2):
    df1['Date'] = pd.to_datetime(df1['Date'], errors='coerce', dayfirst=True)
    df1['Weight'] = df1['Weight'].astype(str).str.replace(',', '.')
    df1['Weight'] = pd.to_numeric(df1['Weight'], errors='coerce')
    df1['Reps'] = pd.to_numeric(df1['Reps'], errors='coerce')
    df1['Sets'] = pd.to_numeric(df1['Sets'], errors='coerce')

    # New columns
    df1['Week'] = df1['Date'].dt.isocalendar().week
    df1['Month'] = df1['Date'].dt.month

    df1 = df1.sort_values(by='Date', ascending=True)

    df2[['kcal', 'kcal total', 'Weight', 'Waist']] = df2[['kcal', 'kcal total', 'Weight', 'Waist']].astype(str).apply(lambda x: x.str.replace(',', '.'))
    df2[['kcal', 'kcal total', 'Weight', 'Waist']] = df2[['kcal', 'kcal total', 'Weight', 'Waist']].apply(pd.to_numeric, errors='coerce')
    df2['Date'] = pd.to_datetime(df2['Date'], format='%d.%m.%Y', dayfirst=True)
    df2['BMI'] = df2['Weight'] / ((height / 100) ** 2) 


    return df1, df2


def all_excercise_volumes(df1, body_weight, outliers, window=8):
    filtered_df = df1[(df1['Muscle group'] != 'Cardio') & (df1['Muscle group'] != 'Walk') & (~df1['Exercise name'].isin(outliers))].copy()

    
    filtered_df.loc[filtered_df['Body weight flg'] == 'BW', 'Weight'] += body_weight
    
    filtered_df['total_reps'] = filtered_df['Reps'] * filtered_df['Sets']
    filtered_df['total_weight'] = filtered_df['Reps'] * filtered_df['Sets'] * filtered_df['Weight']
    
    grouped_df_total = filtered_df.groupby(['Exercise name'])[['Sets', 'Reps', 'Weight', 'total_reps', 'total_weight']].sum()
    grouped_df_total = grouped_df_total.sort_values('total_reps', ascending=False)


    top_exercises = grouped_df_total.head(15)

    filtered_df = filtered_df[filtered_df['Exercise name'].isin(top_exercises.index)]

    grouped_df_sum = filtered_df.groupby(['Date', 'Exercise name'])[['Sets', 'Reps', 'Weight', 'total_reps', 'total_weight']].sum().reset_index()
    grouped_df_mean =  filtered_df.groupby(['Date', 'Exercise name'])[['Sets', 'Reps', 'Weight', 'total_reps', 'total_weight']].mean().reset_index()
    
    grouped_df_sum['total_reps_ma'] = grouped_df_sum.groupby('Exercise name')['total_reps'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    

    unique_exercises = top_exercises.index
    palette = sns.dark_palette(color, len(unique_exercises))
    color_mapping = dict(zip(unique_exercises, palette))

    fig, ((ax3, ax4), (ax1, ax2)) = plt.subplots(2, 2, figsize=(14, 10))

    # Summarizing across all exercises per date
    grouped_df_sum_total = grouped_df_sum.groupby('Date')[['total_weight']].sum().reset_index()
    # grouped_df_sum_total['total_weight'] = grouped_df_sum_total['total_weight'] > 20000
    grouped_df_mean_total = grouped_df_mean.groupby('Date')[['total_weight']].mean().reset_index()
    # grouped_df_mean_total['total_weight'] = grouped_df_mean_total['total_weight'] > 20000
    

    # Plotting the data
    sns.lineplot(data=grouped_df_sum_total, x='Date', y='total_weight', label='Total Weight Sum', ax=ax1, color=color)
    sns.lineplot(data=grouped_df_mean_total, x='Date', y='total_weight', label='Total Weight Mean', ax=ax1, color=color)

    ax1.tick_params(axis='x', rotation=45)
    ax1.set_xlabel('')
    ax1.set_ylabel('Total Reps')
    ax1.set_title('Total Reps Over Time', fontweight='bold')

    sns.barplot(
        data=top_exercises.reset_index(),
        x='Exercise name',
        y='total_reps',
        ax=ax2, 
        hue='Exercise name',
        palette=color_mapping
    )
    
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_xlabel('')
    ax2.set_ylabel('Total Reps')
    ax2.set_title('Total Reps Lifted Over Time', fontweight='bold')
    


    table_data = top_exercises.reset_index()
    table = ax4.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='bottom', bbox=[0, 0.02, 1, 1.34])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(2, 3.5)
    
    
    sns.lineplot(
        ax=ax3,
        data=grouped_df_sum,
        x='Date',
        y='total_reps_ma',
        hue='Exercise name',
        palette=color_mapping, 
        marker='v'
    )
    
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_xlabel('')
    ax3.set_ylabel('Total Reps')
    ax3.set_title('Mean Reps Lifted Over Time', fontweight='bold')

    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor(border_color)
        cell.set_linewidth(0.5)
        if key[0] == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=6)
            cell.set_facecolor('#40466e')
        else:
            cell.set_facecolor('#f2f2f2')

    plt.subplots_adjust(left=0.1, bottom=0.2)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(background_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)

    fig.text(0.95, 0.1, f'Selected outliers: {outliers} were excluded.', ha='right', va='center', fontsize=10, fontweight='normal')
    fig.canvas.manager.set_window_title('Gym Workout Analysis 2024 - Total volumes')
    plt.tight_layout()
    plt.show()

def main() -> None:

    
    df1, df2 = load_data(file_path_1, file_path_2)
    if df1 is not None and df2 is not None:
        df1, df2 = data_preparation(df1, df2)
        all_excercise_volumes(df1, body_weight, outliers)

if __name__ == "__main__":
    main()