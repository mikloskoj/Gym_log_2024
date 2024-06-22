import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

file_path_1 = 'gym_log_Q1_2024 - workout data.csv'
file_path_2 = 'gym_log_Q1_2024 - bio_data.csv'
body_weight = 79
height = 181
sns.set_style("white")
border_color = 'white'
border_color_2 = 'lightgrey'
background_color = '#fdfcfc'
color = "#193f71" # color
false_color = '#611A00'
true_color = '#e0990b'
highlight_color = '#e0990b'
highlight_color_2 = '#f9d48a'
color_palette = sns.dark_palette(color, reverse=True, as_cmap=True)
max_color_value  = '#002347'

line_color = color
title_text_color = '#454545'
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

    start_date = '01.04.2024'
    start_date = pd.to_datetime(start_date, format='%d.%m.%Y', dayfirst=True)
    end_date = df2['Date'].max()
    
    df2_bv = df2
    df2_bv = df2_bv[df2_bv['Date'] >= start_date]
    df2_bv['kcal'] = df2_bv['kcal'].fillna(1)
    df2_bv['kcal total'] = df2_bv['kcal total'].fillna(1)

    fig = plt.figure(constrained_layout=True, figsize=(12, 10))
    gs = gridspec.GridSpec(5, 1, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3:5, 0])  # This subplot spans 2 rows

    sns.lineplot(data=df2_bv, x='Date', y='Waist', color=line_color, ax=ax1)
    ax1.set_title('Waist Over Time', ha='left', fontsize=10, fontweight='bold', color=title_text_color, x=0)
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_xticklabels([])
    ax1.tick_params(axis='y', labelsize=8)

    sns.lineplot(data=df2_bv, x='Date', y='Weight', color=line_color, ax=ax2)
    ax2.set_title('Weight Over Time', ha='left', fontsize=10, fontweight='bold', color=title_text_color, x=0)
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_xticklabels([])
    ax2.tick_params(axis='y', labelsize=8)

    sns.lineplot(data=df2_bv, x='Date', y='BMI', color=line_color, ax=ax3)
    ax3.set_title('BMI Over Time', ha='left', fontsize=10, fontweight='bold', color=title_text_color, x=0)
    ax3.set_xlabel('')
    ax3.set_ylabel('')
    ax3.set_xticklabels([])
    ax3.tick_params(axis='y', labelsize=8)

    # Melt the DataFrame for the barplot
    df_melted = df2_bv.melt(id_vars=['Date'], value_vars=['kcal', 'kcal total'], var_name='Type', value_name='Value')
   
    df_melted_2 = df2_bv.melt(id_vars=['Date'], value_vars=['kcal total'], var_name='Type', value_name='Value')
    df_melted_trshld_min_val = 1800
    df_melted_trshld_upper_val = 2000
    df_melted_trshld_max_val = 2800

    custom_palette = {'kcal': '#f1cb7e', 'kcal total': highlight_color_2}
    sns.barplot(data=df_melted, x='Date', y='Value', hue='Type', palette=custom_palette, ax=ax4, width=0.8, dodge=True, legend=False)

    
    
    # Create custom Line2D objects for the legend
    line1 = Line2D([0], [0], color=highlight_color, linestyle='-', linewidth=3, label=f'{df_melted_trshld_min_val} kcal')
    line2 = Line2D([0], [0], color=false_color, linestyle='-', linewidth=3, label=f'{df_melted_trshld_upper_val} kcal')
    line3 = Line2D([0], [0], color='lightgrey', linestyle='-', linewidth=3, label=f'kcal total')


    # Add the custom lines to the legend
    handles, labels = ax4.get_legend_handles_labels()
    handles.extend([line1, line3])
    labels.extend(['Caloric intake exempt workouts', 'Total kcal intake'])

    ax4.legend(handles=handles, labels=labels, fontsize=8, title_fontsize='10', frameon=False, framealpha=0.9, facecolor='white', edgecolor='lightgrey')
    
    for i, bar in enumerate(ax4.patches):
        value = bar.get_height()
        bar_type = df_melted.iloc[i % len(df_melted)]['Type']
        if value > df_melted_trshld_max_val and df_melted.iloc[i % len(df_melted)]['Type'] == 'kcal':
            bar.set_color(highlight_color)
            bar.set_width(0.8)
            bar.set_edgecolor(None)
            bar.set_linewidth(None)
            bar.set_hatch('')
        elif value > df_melted_trshld_upper_val and df_melted.iloc[i % len(df_melted)]['Type'] == 'kcal':
            bar.set_color(highlight_color_2)
            bar.set_width(0.8)
            bar.set_edgecolor(None)
            bar.set_linewidth(None)
            bar.set_hatch('')
        elif value < df_melted_trshld_min_val and df_melted.iloc[i % len(df_melted)]['Type'] == 'kcal':
            bar.set_color('#e0d3b8')
            bar.set_width(0.8)
            bar.set_edgecolor(None)
            bar.set_linewidth(None)
            bar.set_hatch('')
        elif bar_type == 'kcal total':
            bar.set_color(background_color)
            bar.set_edgecolor('lightgrey')
            bar.set_linewidth(0.2)
            bar.set_hatch('/////')
            


    ax4.set_title('Caloric Intake Over Time', ha='left', fontsize=10, fontweight='bold', color=title_text_color, x=0)
    ax4.set_xlabel('')
    ax4.set_ylabel('')
    ax4.tick_params(axis='x', labelsize=5)
    ax4.tick_params(axis='y', labelsize=8)
    ax4.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    ax4.axhline(df_melted_trshld_min_val, color=highlight_color_2, linestyle='--', linewidth=0.5, alpha=0.5)
    ax4.axhline(df_melted_trshld_upper_val, color=highlight_color, linestyle='--', linewidth=0.5, alpha=0.5)
    ax4.axhline(df_melted_trshld_max_val, color=false_color, linestyle='--', linewidth=0.5, alpha=0.5)
    ax4.set_facecolor('lightblue')
    
    for a in [ax1, ax2, ax3, ax4]:
        a.spines['top'].set_color(border_color)
        a.spines['right'].set_color(border_color)
        a.spines['bottom'].set_color(border_color)
        a.spines['left'].set_color(border_color)
        sns.despine(ax=a, top=False, bottom=False, left=False, right=False)
        a.set_facecolor(background_color)
    fig.suptitle('\nBody Macros View', fontsize=18, color=title_text_color)
    start_date = start_date.strftime('%d.%m.%Y')
    end_date = end_date.strftime('%d.%m.%Y')
    fig.text(0.5, 0.88, f'*This view shows change of body macros over time\nwith calories consumed and burned with workout.\n\nDataframe is in the range from {start_date} to {end_date}', ha='center', fontsize=8, style='italic')
    # plt.subplots_adjust(hspace=0.4, wspace=0)
    plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.95])
    plt.show()


def correlation_view(df2) -> None:
    df_corr_1 = df2[['Waist', 'Weight']].dropna()
    df_corr_1['Waist_MA'] = df_corr_1['Waist'].rolling(window=7).mean()
    df_corr_1['Weight_MA'] = df_corr_1['Weight'].rolling(window=7).mean()
    df_corr_1 = df_corr_1.dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    correlation_matrix = df_corr_1[['Waist_MA', 'Weight_MA']].corr()
    print("Waist vs. Weight Correlation Matrix")
    print(correlation_matrix)

    sns.heatmap(ax=ax1, data=correlation_matrix, annot=True, cmap=color_palette, center=0)
    ax1.set_title("Waist vs. Weight Correlation Matrix", ha='center', fontsize=10, fontweight='bold', color=title_text_color)

    # -------------------------------------------------------

    df_corr_2 = df2[['Weight', 'kcal']].dropna()
    print("Data for correlation_weight_vs_kcal before moving average:\n", df_corr_2.head())  # Debug print

    df_corr_2['Weight_MA'] = df_corr_2['Weight'].rolling(window=7).mean()
    df_corr_2['kcal_MA'] = df_corr_2['kcal'].rolling(window=7).mean()
    df_corr_2 = df_corr_2.dropna()
    print("Data for correlation_weight_vs_kcal after moving average:\n", df_corr_2.head())  # Debug print

    correlation_matrix2 = df_corr_2[['Weight_MA', 'kcal_MA']].corr()
    print("Correlation Matrix for Weight vs. Kcal with Moving Averages:")
    print(correlation_matrix2)

    sns.heatmap(ax=ax2, data=correlation_matrix2, annot=True, cmap=color_palette, center=0)
    ax2.set_title("Weight vs. Kcal Correlation Matrix Heatmap with Moving Averages", ha='center', fontsize=10, fontweight='bold', color=title_text_color)

    fig.suptitle('\nCorrelation view', fontsize=18, color=title_text_color)
    fig.text(0.5, 0.88, '*This view shows correlation between amount of consumed calories and body macros.', ha='center', fontsize=8, style='italic')
    

    plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.95])
    plt.subplots_adjust(hspace=0.1, wspace=0.23)
    plt.gcf().canvas.manager.set_window_title('0X_Correlation view (Gym_log_2024)')
    plt.show()


def day_of_the_week(df1) -> None:
    print(f'Day of the week')    


def sets_view(df1, window=8) -> None:
 
    df_weekly = df1.groupby(['Date', 'Week'])[['Sets']].sum().reset_index()
    df_monthly = df1.groupby(['Date', 'Month'])[['Sets']].sum().reset_index()
    df_daily = df1.groupby(['Date'])[['Sets']].sum().reset_index()
    
    # Create a date range
    date_range = pd.date_range(start=df_daily['Date'].min(), end=df_daily['Date'].max())
    df_daily = df_daily.set_index('Date').reindex(date_range).reset_index()
    df_daily.columns = ['Date', 'Sets']
    print(df_daily.head())
    
    # Create a date range
    date_range = pd.date_range(start=df_weekly['Date'].min(), end=df_weekly['Date'].max())
    df_weekly = df_weekly.set_index('Date').reindex(date_range).reset_index()
    df_weekly.columns = ['Date', 'Week', 'Sets']
    df_weekly = df_weekly.groupby(['Week'])['Sets'].sum().reset_index()
    print(df_weekly.head())
    
    # Create a date range
    date_range = pd.date_range(start=df_monthly['Date'].min(), end=df_monthly['Date'].max())
    df_monthly = df_monthly.set_index('Date').reindex(date_range).reset_index()
    df_monthly.columns = ['Date', 'Month', 'Sets']
    df_monthly = df_monthly.groupby(['Month'])['Sets'].sum().reset_index()
    print(df_monthly.head())

    # get max week
    max_week_idx = df_weekly["Sets"].idxmax()
    max_week = df_weekly.loc[max_week_idx, 'Week']
    print(f'max_week: {max_week}')
    # get max month
    max_month_idx = df_monthly["Sets"].idxmax()
    max_month = df_monthly.loc[max_month_idx, 'Month']
    print(f'max_week: {max_month}')
    # get max day
    max_day_idx = df_daily["Sets"].idxmax()
    max_day = df_daily.loc[max_day_idx, 'Date']
    print(f'max_day: {max_day}')
    
    
    highlight_date_1 = max_week
    highlight_date_2 = max_month
    highlight_date_3 = max_day
    week_colors = [highlight_color if date == highlight_date_1 else
            color for date in df_weekly['Week']]
    month_colors = [highlight_color if date == highlight_date_2 else
            color for date in df_monthly['Month']]
    day_colors = [highlight_color if date == highlight_date_3 else
            'lightgrey' for date in df_daily['Date']]
    
    
    
    
    max_month = df_monthly['Sets']

    fig = plt.figure(figsize=(11, 7))  # Create a figure
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])  # Create a gridspec with specific ratios

    ax0 = plt.subplot(gs[0, 0])  # Top row, first column
    ax1 = plt.subplot(gs[0, 1])  # Top row, second column
    ax2 = plt.subplot(gs[1, :])  # Bottom row, spans both columns

    sns.barplot(
        data=df_monthly.reset_index(),
        x='Month',
        y='Sets', 
        hue='Month',
        palette=month_colors,
        ax=ax0
    )
    ax0.set_title('Monthly Sets', fontsize=14, color=title_text_color, ha='center')
    ax0.legend().remove()
    ax0.set_xlabel('Month',fontsize=6)
    ax0.tick_params(axis='x', labelsize=8)
    ax0.tick_params(axis='y', labelsize=8)

    sns.barplot(
        data=df_weekly.reset_index(),
        x='Week',
        y='Sets',
        hue='Week',
        palette=week_colors,
        ax=ax1
    )
    ax1.set_title('Weekly Sets', fontsize=14, color=title_text_color, ha='center')
    ax1.set_ylabel('')
    ax1.set_xlabel('Week',fontsize=6)
    ax1.legend().remove()
    ax1.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)

    sns.barplot(
        data=df_daily.reset_index(),
        x='Date',
        y='Sets', hue='Date',
        palette=day_colors,
        ax=ax2
    )

    
    '''
            hue='Sets',
        palette=color_palette,
    '''
    
    ax2.set_title('Daily Sets', fontsize=14, color=title_text_color, ha='center')
    ax2.set_ylabel('Sets')
    ax2.set_xlabel('',fontsize=6)
    # ax2.legend().remove()
    ax2.tick_params(axis='x', labelsize=5)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=10))  # Change interval as needed
    # ax2.xticks(rotation=45)

    for ax in [ax0, ax1, ax2]:
        ax.set_facecolor(background_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.suptitle('\nSets Consistency View', fontsize=18, color=title_text_color)
    fig.text(0.5, 0.90, '*Cardio workouts were exempt from the dataset.', ha='center', fontsize=8, style='italic')
    plt.gcf().canvas.manager.set_window_title('0X_Sets view (Gym_log_2024)')
    plt.subplots_adjust(hspace=-0.2, wspace=-0.1)
    plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.95])
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
    ax1.set_title('Total Reps Over Time', ha='left', fontsize=10, fontweight='bold', color=title_text_color, x=0)

    sns.barplot(
        data=top_exercises.reset_index(),
        x='Exercise name',
        y='total_reps',
        ax=ax2, hue='Exercise name',
        palette=color_mapping
    )

    ax2.set_xlabel('')
    ax2.set_ylabel('Total Reps')
    ax2.set_title('Total Reps Lifted Over Time', ha='left', fontsize=10, fontweight='bold', color=title_text_color, x=0)

    table_data = top_exercises.reset_index()
    table = ax2.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='bottom', bbox=[0, -0.5, 1, 0.34])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 0.5)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor(border_color)
        cell.set_linewidth(0.5)
        if key[0] == 0:
            cell.set_text_props(weight='bold', color='grey', fontsize=7)
            cell.set_facecolor('lightgrey')
            cell.set_height(1.5)
        else:
            cell.set_facecolor(background_color)
            cell.set_text_props(weight='normal', color=title_text_color, fontsize=6)
            cell.set_height(0.7)

    plt.subplots_adjust(left=0.1, bottom=0.2)

    for ax in [ax1, ax2]:
        ax.set_facecolor(background_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)

    fig.text(0.95, 0.1, f'Gym Workout Analysis Q1 2024\nThis shows the total volumes I have lifted since 1st of January.\nI hope you like my charts and everything is clear.\nselected_exercises: {selected_exercises} were excluded.', ha='right', va='center', fontsize=10, fontweight='normal')
    plt.gcf().canvas.manager.set_window_title('0X_Workout volumes (Gym_log_2024)')
    
    fig.suptitle('\nExercise Volumes View', fontsize=18, color=title_text_color)
    fig.text(0.5, 0.90, '*Cardio workouts were exempt from the dataset.', ha='center', fontsize=8, style='italic')

    
    plt.subplots_adjust(hspace=-0.2, wspace=-0.1)
    plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.95])
    plt.show()

def consistency_view_table(df1) -> None:

    df1 = df1[(df1['Muscle group'] != 'Cardio') & (df1['Muscle group'] != 'Walk')]
    df1['Duration'] = df1['Duration'].round(1)
    # df1['Reps'] = df1['Reps'].fillna(1)
    df1['Reps'] = df1['Sets'] * df1['Reps']
    df1['Weight'] = df1.apply(lambda row: row['Weight'] + body_weight if row['Body weight flg'] == 'BW' else row['Weight'], axis=1)
    # df1 = df1[(df1['Muscle group'] != 'Cartio') & (df1['Muscle group'] != 'Walk')]
    
    df1 = df1.sort_values(by='Month').reset_index()
    

    df_months = df1.groupby(['Month', 'Month_Name', 'Place'])[['Sets','Reps', 'Weight']].sum().reset_index()
    
    df_muscle = df1.groupby(['Muscle group'])[['Sets', 'Reps', 'Weight']].sum().reset_index()
    df_muscle = df_muscle.sort_values('Sets', ascending=False)
    df_muscle['Sets'] = df_muscle['Sets'].round(1)
    max_sets_muscle_group = df_muscle.iloc[0]['Muscle group']
    
    
    max_reps_muscle_group_df = df_muscle
    max_reps_muscle_group_df = max_reps_muscle_group_df.sort_values('Reps', ascending=False)
    max_reps_muscle_group = max_reps_muscle_group_df.iloc[0]['Muscle group']
    print(f'max_reps_muscle_group {max_reps_muscle_group}')
    
    
    max_weight_muscle_group_df = df_muscle
    max_weight_muscle_group_df = max_weight_muscle_group_df.sort_values('Weight', ascending=False)
    max_weight_muscle_group = max_weight_muscle_group_df.iloc[0]['Muscle group']
    print(f'max_weight_muscle_group {max_weight_muscle_group}')
    
    
    df_place = df1.groupby(['Place'])[['Sets', 'Reps', 'Duration']].sum().reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))


    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(cellText=df_muscle.values, colLabels=df_muscle.columns, cellLoc='center', loc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(8)  # Set font size for better readability
    table1.scale(1.5, 1.2)   # Scale table to fit figure size
    ax1.set_title('Sum of Repetitions by Muscle Group', ha='left', fontsize=10, fontweight='bold', color=title_text_color, x=-0.3, y=0.88)
    

    df_place = df_place.sort_values('Duration', ascending=False)
    
    colors = [highlight_color if place == 'Gym' else 
              color if place == 'Out' else 
              false_color if place == 'Home' else 
            color for place in df_months['Place']]


    sns.barplot(
        ax=ax2,
        data=df_muscle,
        x='Sets',
        y='Muscle group', 
        hue='Sets',
        palette=color_palette
        

    )
    
    

    # Change the color of the confidence interval
    line = ax2.get_lines()[0]
    x_data = line.get_xdata()
    y_data = line.get_ydata()

    # Redraw the confidence interval
    ax2.fill_between(x_data, y_data, color=color, alpha=0)  # Change 'lightblue' to your desired color

    ax2.tick_params(axis='y', labelsize=8)
    ax2.tick_params(axis='x', labelsize=8)
    ax2.set_xlabel('')
    ax2.set_ylabel('Number of hours', fontsize=6)
    ax2.set_title('Sum of Repetitions by Muscle Group chart', ha='left', fontsize=10, fontweight='bold', color=title_text_color, x=-0.3)
    ax2.get_legend().remove()


    
    colors = [highlight_color if place == 'Gym' else 
            color if place == 'Out' else 
            false_color if place == 'Home' else 
            'lightgrey' for place in df_months['Place']]
    


    
    

    for key, cell in table1.get_celld().items():
        cell.set_edgecolor(border_color)
        cell.set_linewidth(0.2)
        cell.set_text_props(weight='normal', color=title_text_color, fontsize=8)
        if key[0] == 0:
            cell.set_text_props(weight='bold', color='grey', fontsize=7)
            cell.set_facecolor('lightgrey')
            cell.set_height(0.15)
        else:
            max_sets_muscle_group_name = df_muscle.iloc[key[0] - 1]['Muscle group']
            max_reps_muscle_group_name = max_reps_muscle_group_df.iloc[key[0] - 1]['Muscle group']
            max_weight_muscle_group_name = max_weight_muscle_group_df.iloc[key[0] - 1]['Muscle group']
            if max_sets_muscle_group_name == max_sets_muscle_group and key[1] == 1:  # Highlight the second column cell
                cell.set_facecolor(highlight_color)
                cell.set_text_props(weight='bold', color='white', fontsize=8)
            elif max_sets_muscle_group_name == max_sets_muscle_group and key[1] == 0:  # Highlight the second column cell
                cell.set_facecolor(max_color_value)
                cell.set_text_props(weight='bold', color='white', fontsize=8)
            elif max_reps_muscle_group_name == max_reps_muscle_group and key[1] == 2:  # Highlight the second column cell
                cell.set_facecolor(background_color)
                cell.set_text_props(weight='bold', color=title_text_color, fontsize=8)
            elif max_weight_muscle_group_name == max_weight_muscle_group and key[1] == 3:  # Highlight the second column cell
                cell.set_facecolor(background_color)
                cell.set_text_props(weight='bold', color=title_text_color, fontsize=8)
            else:
                cell.set_facecolor(background_color)
                cell.set_text_props(weight='normal', color=title_text_color, fontsize=6)
                              

    for ax in [ax1, ax2]:
        ax.set_facecolor(background_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
    
        '''    # Rectangle(xy, width, height, **kwargs)
    rect = patches.Rectangle((0.04, 0.05), 0.9, 0.38, transform=fig.transFigure, linewidth=0.8, edgecolor=border_color_2, facecolor='none')
    fig.patches.append(rect)'''
    
    
    fig.suptitle('\nWorkout Consistency view', fontsize=18, color=title_text_color)
    fig.text(0.5, 0.90, '*Cardio workouts were exempt from the dataset.', ha='center', fontsize=8, style='italic')
    
    plt.gcf().canvas.manager.set_window_title('0X_workout consistency view 2 (Gym_log_2024)')
    plt.subplots_adjust(hspace=-0.2, wspace=-0.1)
    plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.95])
    plt.show()
  
    
    
def consistency_view(df1) -> None:
    
    df1['Duration'] = df1['Duration'].round(1)
    df1['Reps'] = df1['Sets'] * df1['Reps']
    df1['Weight'] = df1.apply(lambda row: row['Weight'] + body_weight if row['Body weight flg'] == 'BW' else row['Weight'], axis=1)
    # df1 = df1[(df1['Muscle group'] != 'Cartio') & (df1['Muscle group'] != 'Walk')]
    
    df1 = df1.sort_values(by='Month').reset_index()

    df_months = df1.groupby(['Month', 'Month_Name', 'Place'])[['Sets','Reps', 'Duration']].sum().reset_index()
    df_months_sets = df1.groupby(['Place'])[['Sets']].sum().reset_index()
    df_months_duration = df1.groupby(['Place'])[['Duration']].sum().reset_index()
    
    df_muscle = df1.groupby(['Muscle group'])[['Duration']].sum().reset_index()
    df_muscle = df_muscle.sort_values('Duration', ascending=False)
    df_muscle['Duration'] = df_muscle['Duration'].round(1)
    df_place = df1.groupby(['Place'])[['Sets', 'Reps', 'Duration']].sum().reset_index()
  

    
    print(df_months.head(5))
    print(df_place.head(5))


    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    # plt.subplots_adjust(hspace=-0.2, wspace=-0.1)
    fig.suptitle('\nWorkout Consistency Hourly View', fontsize=18, color=title_text_color)

    # First table
    colors = [highlight_color if place == 'Gym' else 
            color if place == 'Out' else 
            false_color if place == 'Home' else 
            'lightgrey' for place in df_months['Place']]
    
    sns.barplot(
        ax=ax1,
        data=df_months,
        x='Month_Name',
        y='Duration', 
        palette=colors, 
        hue='Place'

    )
    
    ax1.tick_params(axis='y', labelsize=8)
    ax1.tick_params(axis='x', labelsize=8)
    pos1 = ax1.get_position()  # Get the original position of ax4
    pos2 = [pos1.x0, pos1.y0 + 0.11, pos1.width, pos1.height]  # Modify the position
    ax1.set_position(pos2)  # Set the new position
    # ax2.set_title('Workout Sessions by Place')
    # ax4.get_legend().remove()
    ax1.set_xlabel('')
    ax1.set_ylabel('Number of hours', fontsize=6)
    # fig.text(0.05, 0.88, 'Consistency each month and grouped by place', ha='left', fontsize=11, fontweight='bold', color=title_text_color)
    
    # Rectangle(xy, width, height, **kwargs)
    # rect = patches.Rectangle((0.04, 0.05), 0.9, 0.38, transform=fig.transFigure, linewidth=0.8, edgecolor=border_color_2, facecolor='none')
    # fig.patches.append(rect)


    df_place = df_place.sort_values('Duration', ascending=False)
    
    colors = [highlight_color if place == 'Gym' else 
              color if place == 'Out' else 
              false_color if place == 'Home' else 
            color for place in df_months['Place']]


    sns.lineplot(
        ax=ax2,
        data=df_months,
        x='Month_Name',
        y='Duration', 
        hue='Place',
        marker='v',
        palette=colors,
        linestyle= '-',
        linewidth=0.8
        

    )
    
    

    # Change the color of the confidence interval
    line = ax2.get_lines()[0]
    x_data = line.get_xdata()
    y_data = line.get_ydata()

    # Redraw the confidence interval
    ax2.fill_between(x_data, y_data, color=color, alpha=0)  # Change 'lightblue' to your desired color

    ax2.tick_params(axis='y', labelsize=8)
    ax2.tick_params(axis='x', labelsize=8)
    ax2.set_xlabel('')
    ax2.set_ylabel('Number of hours', fontsize=6)


    # Second table
    ax3.axis('tight')
    ax3.axis('off')
    table2 = ax3.table(cellText=df_place.values, colLabels=df_place.columns, cellLoc='center', loc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)  # Set font size for better readability
    table2.scale(1, 1.2)   # Scale table to fit figure size
    # fig.text(0.05, 0.45, 'Place of workout view', ha='left', fontsize=11, fontweight='bold', color=title_text_color)
    
    colors = [highlight_color if place == 'Gym' else 
            color if place == 'Out' else 
            false_color if place == 'Home' else 
            'lightgrey' for place in df_months['Place']]
    
    sns.barplot(
        ax=ax4,
        data=df_months_duration,
        x='Place',
        y='Duration', 
        palette=colors, 
        hue='Place'

    )
    
    ax4.tick_params(axis='y', labelsize=8)
    ax4.tick_params(axis='x', labelsize=8)
    pos1 = ax4.get_position()  # Get the original position of ax4
    pos2 = [pos1.x0, pos1.y0 + 0.11, pos1.width, pos1.height]  # Modify the position
    ax4.set_position(pos2)  # Set the new position
    # ax2.set_title('Workout Sessions by Place')
    # ax4.get_legend().remove()
    ax4.set_xlabel('')
    ax4.set_ylabel('Number of hours', fontsize=6)

                
    for key, cell in table2.get_celld().items():
        cell.set_edgecolor(border_color)
        cell.set_linewidth(0.2)
        cell.set_text_props(weight='normal', color=title_text_color, fontsize=8)
        cell.set_height(0.2)
        if key[0] == 0:
            cell.set_text_props(weight='bold', color='grey', fontsize=7)
            cell.set_facecolor('lightgrey')
            cell.set_height(0.1)
        else:
            cell.set_facecolor('#f2f2f2')
            # Retrieve the text from the cell
            cell_text = cell.get_text().get_text()
            cell.set_text_props(weight='normal', color=title_text_color, fontsize=6)
            
            # Check if the cell contains a certain text (e.g., "Important")
            if "Gym" in cell_text:
                cell.set_facecolor(highlight_color)  # Change background color for cells containing "Important"
                cell.set_text_props(color='white', weight='bold')  # Change text color for cells containing "Important"
            elif "Out" in cell_text:
                cell.set_facecolor(color)  # Change background color for cells containing "Important"
                cell.set_text_props(color='white', weight='bold')  # Change text color for cells containing "Important"    
            elif "Home" in cell_text:
                cell.set_facecolor(false_color)  # Change background color for cells containing "Important"
                cell.set_text_props(color='white', weight='bold')  # Change text color for cells containing "Important"                

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(background_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
    # Display the tables
    plt.gcf().canvas.manager.set_window_title('0X_workout consistency view 2  (Gym_log_2024)')
    fig.text(0.5, 0.90, '*Data collected from January to June 2024', ha='center', fontsize=8, style='italic')   
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    # plt.tight_layout()

    plt.show()

def workout_details(df1) -> None:
    
    df1 = df1[(df1['Muscle group'] != 'Cardio') & (df1['Muscle group'] != 'Walk')]
    df1['Reps'] = df1['Reps'].fillna(1)
    df1['Total_reps'] = df1['Sets'] * df1['Reps']


    # Group by 'Date' and sum 'Total_reps'
    df1 = df1.groupby(['Date', 'Week'])['Total_reps'].sum().reset_index()

    # Create a date range
    date_range = pd.date_range(start=df1['Date'].min(), end=df1['Date'].max())

    # Reindex the dataframe to include all dates in the range
    df1 = df1.set_index('Date').reindex(date_range).reset_index()
    df1.columns = ['Date', 'Week', 'Total_reps']

    average_initial_reps = df1["Total_reps"].head(30).mean()
    average_final_reps = df1["Total_reps"].tail(30).mean()

    avg = df1["Total_reps"].mean()
    print(f'AVG: {avg}')
    max = df1["Total_reps"].max()
    print(f'MAX: {max}')

    max_idx = df1["Total_reps"].idxmax()
    max_date = df1.loc[max_idx, 'Date']
    min_idx = df1["Total_reps"].idxmin()
    min_date = df1.loc[min_idx, 'Date']
    print(f'max_date: {max_date}')



    # Highlight specific dates
    highlight_date_1 = pd.Timestamp('2024-01-14')
    highlight_date_2 = pd.Timestamp('2024-04-29')
    highlight_date_3 = max_date
    highlight_date_4 = min_date
    colors = ['lightgrey' if date == highlight_date_1 else 
            'lightgrey' if date == highlight_date_2 else 
            highlight_color if date == highlight_date_3 else 
            false_color if date == highlight_date_4 else 
            'lightgrey' for date in df1['Date']]

    # Plotting
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Create barplot
    sns.barplot(ax=ax1, x=df1['Date'], y=df1['Total_reps'], hue=df1['Date'], palette=colors, width=1)
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))  # Change interval as needed
    plt.xticks(rotation=45)

    # Set labels and title
    ax1.set_xlabel('')
    ax1.set_ylabel('Total reps', color=title_text_color, fontsize=10, weight='bold')
    ax1.set_title(f'Total reps over time', color=title_text_color, weight='bold', fontsize=16, loc='left')
    ax1.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)

    # Set the background color
    ax1.set_facecolor(background_color)

    # Create custom legend
    
    legend_handles = [
        Patch(color=highlight_color, label='Max'),
        Patch(color=false_color, label='Min')
    ]
    ax1.legend(handles=legend_handles, title='', edgecolor=border_color_2)

    # Define the inset plot
    inset_ax = inset_axes(ax1, width="25%", height="15%", loc="lower left")  # Adjust width, height, and loc as needed

    # Text plot data
    text = f"Mean change in Total Reps:\non average"
    inset_ax.text(0.05, 0.9, text, fontsize=8, color=title_text_color, ha='left', va='top')
    inset_ax.set_xticks([])  # Remove x-ticks
    inset_ax.set_yticks([])  # Remove y-ticks
    inset_ax.patch.set_facecolor('white')
    inset_ax.patch.set_alpha(0.85)




    # df['Total_reps'] = df['Total_reps'].dropna()
    # df = df.dropna(subset=['Date'])

    sns_regplot = sns.regplot(x=np.arange(0, len(df1['Date'])), y=df1['Total_reps'], ax=ax1, marker='', color=color ,line_kws={
            'color': highlight_color,  # Line color
            'linestyle': ':',  # Line style (e.g., '--' for dashed line)
            'linewidth': 1  # Line width
        })


    df_x = df1[['Week', 'Total_reps']]
    df_x = df_x.dropna(subset=['Week'])
    df_x = df_x.groupby('Week')['Total_reps'].sum().reset_index()
    df_x['Change'] = df_x['Total_reps'].pct_change() * 100
    df_x = df_x.groupby('Week')['Change'].sum().reset_index()
    print(df_x.head(30))


    df_x['Positive'] = df_x['Change'] >= 0
    # Define the inset plot
    inset_ax2 = inset_axes(ax1, width="30%", height="15%", loc="upper left")  # Adjust width, height, and loc as needed

    # Text plot data
    text = sns.barplot(data=df_x, y=df_x['Change'], x=df_x['Week'], hue='Positive',palette={True: true_color, False: false_color}, dodge=False)
    inset_ax2.set_xlabel('')
    inset_ax2.set_ylabel('')
    inset_ax2.tick_params(axis='x', direction='in', labelsize=7, color=title_text_color)
    inset_ax2.set_yticks([])
    inset_ax2.set_title('')
    inset_ax2.legend().remove()
    
    # Add custom title inside the axes
    inset_ax2.text(
        0.05, 0.92, 'Weekly change in sets volume',
        horizontalalignment='left',
        verticalalignment='top',
        transform=inset_ax2.transAxes,
        weight='normal',
        fontsize=9,
        color=title_text_color
    )
    
    inset_ax2.patch.set_facecolor('white')
    inset_ax2.patch.set_alpha(0.85)


    for spine in ax1.spines.values():
        spine.set_edgecolor('white')
        
    for spine in inset_ax.spines.values():
        spine.set_edgecolor(border_color_2)
    
    for spine in inset_ax2.spines.values():
        spine.set_edgecolor(border_color_2)



    # inset_ax2.set_yticks([])  # Remove y-ticks
    # inset_ax2.axhline(0, color="r", clip_on=False)


    plt.gcf().canvas.manager.set_window_title('0X_Workout details  (Gym_log_2024)')
    # Show plot
    plt.show()

def main():
    df1, df2 = load_data(file_path_1, file_path_2)
    if df1 is not None and df2 is not None:
        df1, df2 = data_preparation(df1, df2)

 

        
        excercise_volumes(df1, body_weight, selected_exercises)    
               
        '''
        workout_details(df1) # DONE         

        consistency_view(df1)    
        consistency_view_table(df1) # DONE  
        sets_view(df1)  
        body_values(df2) # DONE  
        
        day_of_the_week(df1) 
        correlation_view(df2)   
       

        '''


if __name__ == "__main__":
    main()
