import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    week_colors = ['orange' if date == highlight_date_1 else
            color for date in df_weekly['Week']]
    month_colors = ['orange' if date == highlight_date_2 else
            color for date in df_monthly['Month']]
    day_colors = ['orange' if date == highlight_date_3 else
            color for date in df_daily['Date']]
    
    
    
    
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
    ax0.set_title('Monthly Sets', fontweight='bold', fontsize=12)
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
    ax1.set_title('Weekly Sets', fontweight='bold', fontsize=12)
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
    
    ax2.set_title('Daily Sets', fontweight='bold', fontsize=12)
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


    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=-0.2, wspace=-0.1)
    fig.suptitle('Workout Consistency view', fontsize=18, color=title_text_color)

    # First table
    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(cellText=df_months.values, colLabels=df_months.columns, cellLoc='center', loc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(8)  # Set font size for better readability
    table1.scale(1.5, 1.2)   # Scale table to fit figure size
    fig.text(0.05, 0.9, 'Consistency each month and grouped by place', ha='left', fontsize=11, fontweight='bold', color=title_text_color)
    
    # Rectangle(xy, width, height, **kwargs)
    rect = patches.Rectangle((0.04, 0.05), 0.9, 0.38, transform=fig.transFigure, linewidth=0.8, edgecolor=title_text_color, facecolor='none')
    fig.patches.append(rect)

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
    fig.text(0.05, 0.45, 'Place of workout view', ha='left', fontsize=11, fontweight='bold', color=title_text_color)
    
    sns.barplot(
        ax=ax4,
        data=df_place,
        x='Place',
        y='Duration', color=color

    )
    
    ax4.tick_params(axis='y', labelsize=8)
    ax4.tick_params(axis='x', labelsize=8)
    pos1 = ax4.get_position()  # Get the original position of ax4
    pos2 = [pos1.x0, pos1.y0 + 0.11, pos1.width, pos1.height]  # Modify the position
    ax4.set_position(pos2)  # Set the new position
    # ax2.set_title('Workout Sessions by Place')

    plt.tight_layout(rect=[0, 0.1, 0.95, 0.95])
    fig.text(0.2, 0.01, 'Data collected from January to June 2024', ha='center', fontsize=8, style='italic')

    for key, cell in table1.get_celld().items():
            cell.set_edgecolor(border_color)
            cell.set_linewidth(0.2)
            cell.set_text_props(weight='normal', color=title_text_color, fontsize=8)
            if key[0] == 0:
                cell.set_text_props(weight='bold', color='white', fontsize=7)
                cell.set_facecolor('#40466e')
                cell.set_height(0.1)
            else:
                cell.set_facecolor('#f2f2f2')
                
    for key, cell in table2.get_celld().items():
        cell.set_edgecolor(border_color)
        cell.set_linewidth(0.2)
        cell.set_text_props(weight='normal', color=title_text_color, fontsize=8)
        cell.set_height(0.2)
        if key[0] == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=7)
            cell.set_facecolor('#40466e')
            cell.set_height(0.1)
        else:
            cell.set_facecolor('#f2f2f2')

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(background_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
    # Display the tables
    plt.show()

def workout_details(df1) -> None:
    
    '''
    df1['Date'] = pd.to_datetime(df1['Date'], format='%d.%m.%Y')
    df1['Week'] = df1['Date'].dt.isocalendar().week

    columns_to_convert = ['Sets', 'Reps', 'Weight', 'Duration']
    df1[columns_to_convert] = df1[columns_to_convert].astype(str).apply(lambda x: x.str.replace(',', '.'))
    df1[columns_to_convert] = df1[columns_to_convert].apply(lambda x: pd.to_numeric(x, errors='coerce'))
'''
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
    colors = ['red' if date == highlight_date_1 else 
            'orange' if date == highlight_date_2 else 
            'green' if date == highlight_date_3 else 
            'darkblue' if date == highlight_date_4 else 
            'lightgrey' for date in df1['Date']]

    # Plotting
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Create barplot
    sns.barplot(ax=ax1, x=df1['Date'], y=df1['Total_reps'], hue=df1['Date'], palette=colors)
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))  # Change interval as needed
    plt.xticks(rotation=45)

    # Set labels and title
    ax1.set_xlabel('')
    ax1.set_ylabel('Total_reps')
    ax1.set_title(f'Total reps over time')

    # Set the background color
    ax1.set_facecolor(background_color)

    # Create custom legend
    legend_handles = [
        Patch(color='red', label='2024-01-14'),
        Patch(color='orange', label='2024-04-29'),
        Patch(color='lightgrey', label='Other Dates')
    ]
    ax1.legend(handles=legend_handles, title='Legend')

    # Define the inset plot
    inset_ax = inset_axes(ax1, width="30%", height="15%", loc="upper left")  # Adjust width, height, and loc as needed

    # Text plot data
    text = f"Mean change in Total Reps:\n on average"
    inset_ax.text(0.5, 0.5, text, fontsize=12, ha='center')
    inset_ax.set_xticks([])  # Remove x-ticks
    inset_ax.set_yticks([])  # Remove y-ticks


    # Set window title
    plt.gcf().canvas.manager.set_window_title('02_Gym Log Data Visualization')

    # df['Total_reps'] = df['Total_reps'].dropna()
    # df = df.dropna(subset=['Date'])

    sns_regplot = sns.regplot(x=np.arange(0, len(df1['Date'])), y=df1['Total_reps'], ax=ax1, marker='.', color='orange',line_kws={
            'color': 'orange',  # Line color
            'linestyle': '-',  # Line style (e.g., '--' for dashed line)
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
    inset_ax2 = inset_axes(ax1, width="30%", height="15%", loc="lower left")  # Adjust width, height, and loc as needed

    # Text plot data
    text = sns.barplot(data=df_x, y=df_x['Change'], x=df_x['Week'], hue='Positive',palette={True: 'green', False: 'red'}, dodge=False)
    inset_ax2.set_xticks([])  # Remove x-ticks
    inset_ax2.set_title('Weekly change in sets volume', weight='normal', loc='right', fontsize=9, color='black')
    inset_ax2.legend().remove()
    inset_ax2.patch.set_facecolor('white')
    inset_ax2.patch.set_alpha(0.7)

    for spine in ax1.spines.values():
        spine.set_edgecolor(border_color)
    
    for spine in inset_ax2.spines.values():
        spine.set_edgecolor('lightgrey')

    # inset_ax2.set_yticks([])  # Remove y-ticks
    # inset_ax2.axhline(0, color="r", clip_on=False)

    # Show plot
    plt.show()

def main():
    df1, df2 = load_data(file_path_1, file_path_2)
    if df1 is not None and df2 is not None:
        df1, df2 = data_preparation(df1, df2)

        workout_details(df1)
        consistency_view(df1)
        body_values(df2)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        correlation_waist_v_weight(df2, axes[0])
        correlation_weight_vs_kcal(df2, axes[1])
        plt.show()
        
        excercise_volumes(df1, body_weight, selected_exercises)
        sets_view(df1)
        '''
        
       

        '''


if __name__ == "__main__":
    main()
