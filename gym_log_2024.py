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
border_color = 'white'
background_color = '#fdfcfc'
color = "#193f71" # color
color_palette = sns.dark_palette(color, reverse=True, as_cmap=True)
line_color = color
selected_exercises = ('Kneeling dip', 'Bench press', 'Chest press', 'Prone leg curl', 'Lat pulldown', 'Bicep curl')


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
    
    columns_to_convert = ['Reps', 'Sets', 'Weight', 'Duration']
    df1[columns_to_convert] = df1[columns_to_convert].astype(str).apply(lambda x: x.str.replace(',','.'))
    df1[columns_to_convert] = df1[columns_to_convert].astype(str).apply(lambda x: pd.to_numeric(x, errors='coerce'))
    df1['Date'] = pd.to_datetime(df1['Date'], format='%d.%m.%Y' , errors='coerce')
    df1['Month'] = df1['Date'].dt.month
    df1['Week'] = df1['Date'].dt.isocalendar().week
    df1['Month_Name'] = df1['Date'].dt.strftime('%B')
    df1 = df1.sort_values(by='Date', ascending=True)
    

    df2[['kcal', 'kcal total', 'Weight', 'Waist']] = df2[['kcal', 'kcal total', 'Weight', 'Waist']].astype(str).apply(lambda x: x.str.replace(',', '.'))
    df2[['kcal', 'kcal total', 'Weight', 'Waist']] = df2[['kcal', 'kcal total', 'Weight', 'Waist']].apply(pd.to_numeric, errors='coerce')
    df2['Date'] = pd.to_datetime(df2['Date'], format='%d.%m.%Y', dayfirst=True)
    df2['BMI'] = df2['Weight'] / ((height / 100) ** 2) 
    
    



    return df1, df2

def body_values(df2):

    start_date = '2024-04-01'
    start_date = pd.to_datetime(start_date)
    
    df2_bv = df2
    df2_bv = df2_bv[df2_bv['Date'] >= start_date]


    # Apply moving average to smooth the data
    df2_bv.loc[:, 'Weight'] = df2_bv['Weight'].rolling(window=1).mean()
    df2_bv.loc[:, 'Waist'] = df2_bv['Waist'].rolling(window=1).mean()
    df2_bv.loc[:, 'BMI'] = df2_bv['BMI'].rolling(window=1).mean()

    # Create a figure and axis
    fig, ax = plt.subplots(4, 1, figsize=(10, 8))

    sns.lineplot(data=df2_bv, x='Date', y='Waist', color=line_color, ax=ax[0])
    ax[0].set_title('Waist Over Time', fontweight='bold', fontsize=12, loc='left')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('')
    ax[0].set_xticklabels([])
    ax[0].tick_params(axis='y', labelsize=8)

    sns.lineplot(data=df2_bv, x='Date', y='Weight', color=line_color, ax=ax[1])
    ax[1].set_title('Weight Over Time', fontweight='bold', fontsize=12, loc='left')
    ax[1].set_xlabel('')
    ax[1].set_ylabel('')
    ax[1].set_xticklabels([])
    ax[1].tick_params(axis='y', labelsize=8)

    sns.lineplot(data=df2_bv, x='Date', y='BMI', color=line_color, ax=ax[2])
    ax[2].set_title('BMI Over Time', fontweight='bold', fontsize=12, loc='left')
    ax[2].set_xlabel('')
    ax[2].set_ylabel('')
    ax[2].set_xticklabels([])
    ax[2].tick_params(axis='y', labelsize=8)

    # Melt the DataFrame for the barplot
    df_melted = df2_bv.melt(id_vars=['Date'], value_vars=['kcal', 'kcal total'], var_name='Type', value_name='Value')
    custom_palette = {'kcal': line_color, 'kcal total': 'lightgrey'}
    
    sns.barplot(data=df_melted, x='Date', y='Value', hue='Type', palette=custom_palette, ax=ax[3])
    ax[3].set_title('Caloric Intake Over Time', fontweight='bold', fontsize=12, loc='left')
    ax[3].set_xlabel('')
    ax[3].set_ylabel('')
    ax[3].set_xticklabels([])
    ax[3].tick_params(axis='y', labelsize=8)
    ax[3].axhline(1800, color='#112c50', linestyle='--', linewidth=1)
    ax[3].axhline(2000, color='#112c50', linestyle='--', linewidth=1)
    
    # Create custom Line2D objects for the legend
    line1 = Line2D([0], [0], color='#112c50', linestyle='--', linewidth=1, label='1800 Cal')
    line2 = Line2D([0], [0], color='#112c50', linestyle='--', linewidth=1, label='2000 Cal')

    # Add the custom lines to the legend
    handles, labels = ax[3].get_legend_handles_labels()
    handles.extend([line1, line2])
    labels.extend(['lower limit', 'upper limit'])
    
    ax[3].legend(handles=handles, labels=labels, fontsize=8, title_fontsize='10', frameon=True, framealpha=0.9, facecolor='white', edgecolor='black')

    # legend = ax[3].legend(loc='upper left', title='Caloric Type', fontsize=8, title_fontsize='10', frameon=True, framealpha=0.9, facecolor='white', edgecolor='black')
    # legend.set_bbox_to_anchor((0, 1))

    for a in ax[0:]:
        a.spines['top'].set_color(border_color)
        a.spines['right'].set_color(border_color)
        a.spines['bottom'].set_color(border_color)
        a.spines['left'].set_color(border_color)
        sns.despine(ax=a, top=False, bottom=False, left=False, right=False)
        a.set_facecolor(background_color)

    plt.tight_layout()
    plt.show()

def correlation_waist_v_weight(df2, ax) -> None:
    df_corr_1 = df2[['Waist', 'Weight']].dropna()
    df_corr_1['Waist_MA'] = df_corr_1['Waist'].rolling(window=7).mean()
    df_corr_1['Weight_MA'] = df_corr_1['Weight'].rolling(window=7).mean()
    df_corr_1 = df_corr_1.dropna()

    correlation_matrix = df_corr_1[['Waist_MA', 'Weight_MA']].corr()
    print("Waist vs. Weight Correlation Matrix")
    print(correlation_matrix)

    sns.heatmap(correlation_matrix, annot=True, cmap=color_palette, center=0, ax=ax)
    ax.set_title("Waist vs. Weight Correlation Matrix")

def correlation_weight_vs_kcal(df2, ax) -> None:
    df_corr_2 = df2[['Weight', 'kcal']].dropna()
    print("Data for correlation_weight_vs_kcal before moving average:\n", df_corr_2.head())  # Debug print

    df_corr_2['Weight_MA'] = df_corr_2['Weight'].rolling(window=7).mean()
    df_corr_2['kcal_MA'] = df_corr_2['kcal'].rolling(window=7).mean()
    df_corr_2 = df_corr_2.dropna()
    print("Data for correlation_weight_vs_kcal after moving average:\n", df_corr_2.head())  # Debug print

    correlation_matrix = df_corr_2[['Weight_MA', 'kcal_MA']].corr()
    print("Correlation Matrix for Weight vs. Kcal with Moving Averages:")
    print(correlation_matrix)

    sns.heatmap(correlation_matrix, annot=True, cmap=color_palette, center=0, ax=ax)
    ax.set_title("Weight vs. Kcal Correlation Matrix Heatmap with Moving Averages")

def sets_view(df1, window=8) -> None:
    df_weekly = df1.groupby(['Week'])[['Sets']].sum()
    df_monthly = df1.groupby(['Month'])[['Sets']].sum()
    df_daily = df1.groupby(['Date'])[['Sets']].sum()
   

    print(df_daily.head())

    fig = plt.figure(figsize=(11, 7))  # Create a figure
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])  # Create a gridspec with specific ratios

    ax0 = plt.subplot(gs[0, 0])  # Top row, first column
    ax1 = plt.subplot(gs[0, 1])  # Top row, second column
    ax2 = plt.subplot(gs[1, :])  # Bottom row, spans both columns

    sns.barplot(
        data=df_monthly.reset_index(),
        x='Month',
        y='Sets',
        hue='Sets',
        palette=color_palette,
        ax=ax0
    )
    ax0.set_title('Monthly Sets', fontweight='bold', fontsize=12)
    ax0.legend().remove()
    ax0.set_xlabel('Month',fontsize=6)
    ax0.tick_params(axis='x', labelsize=8)
    ax0.tick_params(axis='y', labelsize=8)

    sns.barplot(
        data=df_weekly.reset_index(),
        x='Week',
        y='Sets',
        hue='Sets',
        palette=color_palette,
        ax=ax1
    )
    ax1.set_title('Weekly Sets', fontweight='bold', fontsize=12)
    ax1.set_ylabel('')
    ax1.set_xlabel('Week',fontsize=6)
    ax1.legend().remove()
    ax1.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)

    sns.barplot(
        data=df_daily.reset_index(),
        x='Date',
        y='Sets',
        hue='Sets',
        palette=color_palette,
        ax=ax2
    )
    ax2.set_title('Daily Sets', fontweight='bold', fontsize=12)
    ax2.set_ylabel('')
    ax2.set_xlabel('Day',fontsize=6)
    ax2.legend().remove()
    ax2.tick_params(axis='x', labelsize=5)
    ax2.tick_params(axis='y', labelsize=8)

    for ax in [ax0, ax1, ax2]:
        ax.set_facecolor(background_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def excercise_volumes(df1, body_weight, selected_exercises, window=8):
    filtered_df = df1[(df1['Muscle group'] != 'Cardio') & (df1['Muscle group'] != 'Walk') & (df1['Exercise name'].isin(selected_exercises))].copy()

    
    filtered_df.loc[filtered_df['Body weight flg'] == 'BW', 'Weight'] += body_weight
    
    filtered_df['total_reps'] = filtered_df['Reps'] * filtered_df['Sets']
    filtered_df['total_weight'] = filtered_df['Reps'] * filtered_df['Sets'] * filtered_df['Weight']
    
    grouped_df_total = filtered_df.groupby(['Exercise name'])[['Sets', 'Reps', 'Weight', 'total_reps', 'total_weight']].sum()
    grouped_df_total = grouped_df_total.sort_values('total_reps', ascending=False)

    top_exercises = grouped_df_total.head(5)

    filtered_df = filtered_df[filtered_df['Exercise name'].isin(top_exercises.index)]

    grouped_df = filtered_df.groupby(['Date', 'Exercise name'])[['Sets', 'Reps', 'Weight', 'total_reps', 'total_weight']].sum().reset_index()
    grouped_df['total_reps_ma'] = grouped_df.groupby('Exercise name')['total_reps'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

    unique_exercises = top_exercises.index
    palette = sns.dark_palette(color, len(unique_exercises))
    color_mapping = dict(zip(unique_exercises, palette))

    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(14, 7))

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
    ax1.set_xlabel('')
    ax1.set_ylabel('Total Reps')
    ax1.set_title('Total Reps Over Time', fontweight='bold')

    sns.barplot(
        data=top_exercises.reset_index(),
        x='Exercise name',
        y='total_reps',
        ax=ax2, hue='Exercise name',
        palette=color_mapping
    )

    ax2.set_xlabel('')
    ax2.set_ylabel('Total Reps')
    ax2.set_title('Total Reps Lifted Over Time', fontweight='bold')

    table_data = top_exercises.reset_index()
    table = ax2.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='bottom', bbox=[0, -0.5, 1, 0.34])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 0.5)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor(border_color)
        cell.set_linewidth(0.5)
        if key[0] == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=6)
            cell.set_facecolor('#40466e')
        else:
            cell.set_facecolor('#f2f2f2')

    plt.subplots_adjust(left=0.1, bottom=0.2)

    for ax in [ax1, ax2]:
        ax.set_facecolor(background_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)

    fig.text(0.95, 0.1, f'Gym Workout Analysis Q1 2024\nThis shows the total volumes I have lifted since 1st of January.\nI hope you like my charts and everything is clear.\nselected_exercises: {selected_exercises} were excluded.', ha='right', va='center', fontsize=10, fontweight='normal')
    fig.canvas.manager.set_window_title('Gym Workout Analysis 2024 - Total volumes')
    plt.tight_layout()
    plt.show()

def consistency_view(df1) -> None:
    
    df1['Duration'] = df1['Duration'].round(1)
    df1['Reps'] = df1['Sets'] * df1['Reps']
    df1['Weight'] = df1.apply(lambda row: row['Weight'] + body_weight if row['Body weight flg'] == 'BW' else row['Weight'], axis=1)

    df1 = df1.sort_values(by='Month').reset_index()



    df_months = df1.groupby(['Month', 'Month_Name', 'Place'])[['Sets','Reps','Duration']].sum().reset_index()
    df_place = df1.groupby(['Place'])[['Sets', 'Reps', 'Duration']].sum().reset_index()

    print(df_months.head(5))
    print(df_place.head(5))


    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
    fig.suptitle('Gym Workout Data', fontsize=16)

    # First table
    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(cellText=df_months.values, colLabels=df_months.columns, cellLoc='center', loc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(8)  # Set font size for better readability
    table1.scale(1.5, 1.2)   # Scale table to fit figure size

    sns.lineplot(
        ax=ax2,
        data=df_months,
        x='Month',
        y='Duration',
        color= line_color

    )
    ax2.tick_params(axis='y', labelsize=8)
    ax2.tick_params(axis='x', labelsize=8)

    # Second table
    ax3.axis('tight')
    ax3.axis('off')
    table2 = ax3.table(cellText=df_place.values, colLabels=df_place.columns, cellLoc='center', loc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)  # Set font size for better readability
    table2.scale(1, 1.2)   # Scale table to fit figure size
    
    sns.barplot(
        ax=ax4,
        data=df_place,
        x='Place',
        y='Duration', color=color

    )
    
    ax4.tick_params(axis='y', labelsize=8)
    ax4.tick_params(axis='x', labelsize=8)
    # ax2.set_title('Workout Sessions by Place')


    # Adjust layout to ensure everything fits well
    plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.text(0.2, 0.01, 'Data collected from January to June 2024', ha='center', fontsize=8, style='italic')

    for key, cell in table1.get_celld().items():
            cell.set_edgecolor(border_color)
            cell.set_linewidth(0.2)
            if key[0] == 0:
                cell.set_text_props(weight='bold', color='white', fontsize=6)
                cell.set_facecolor('#40466e')
            else:
                cell.set_facecolor('#f2f2f2')
                
    for key, cell in table2.get_celld().items():
        cell.set_edgecolor(border_color)
        cell.set_linewidth(0.2)
        if key[0] == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=6)
            cell.set_facecolor('#40466e')
        else:
            cell.set_facecolor('#f2f2f2')

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(background_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
    # Display the tables
    plt.show()


def main():
    df1, df2 = load_data(file_path_1, file_path_2)
    if df1 is not None and df2 is not None:
        df1, df2 = data_preparation(df1, df2)
        consistency_view(df1)

        '''
                body_values(df2)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        correlation_waist_v_weight(df2, axes[0])
        correlation_weight_vs_kcal(df2, axes[1])
        plt.show()
        sets_view(df1)
        excercise_volumes(df1, body_weight, selected_exercises)
       

        '''


if __name__ == "__main__":
    main()
