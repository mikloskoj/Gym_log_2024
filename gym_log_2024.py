import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.gridspec as gridspec

file_path_1 = r'C:\Users\janmi\Documents\VS Code\Gym_log_2024\gym_log_Q1_2024 - workout data.csv'
file_path_2 = r'C:\Users\janmi\Documents\VS Code\Gym_log_2024\gym_log_Q1_2024 - bio_data.csv'
body_weight = 79
height = 181
sns.set_style("white")
border_color = 'lightgrey'
background_color = '#fdfcfc'
color_palette = sns.dark_palette("#69d", reverse=True, as_cmap=True)
line_color = '#173b6a'
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
    df1['Date'] = pd.to_datetime(df1['Date'], errors='coerce', dayfirst=True)
    df1['Weight'] = df1['Weight'].str.replace(',', '.')
    df1['Weight'] = pd.to_numeric(df1['Weight'], errors='coerce')
    df1['Reps'] = pd.to_numeric(df1['Reps'], errors='coerce')
    df1['Sets'] = pd.to_numeric(df1['Sets'], errors='coerce')

    # New columns
    df1['Week'] = df1['Date'].dt.isocalendar().week
    df1['Month'] = df1['Date'].dt.month

    df1 = df1.sort_values(by='Date', ascending=True)

    df2[['kcal', 'kcal Total', 'Wgt (kg)', 'Waist (cm)']] = df2[['kcal', 'kcal Total', 'Wgt (kg)', 'Waist (cm)']].astype(str).apply(lambda x: x.str.replace(',', '.'))
    df2[['kcal', 'kcal Total', 'Wgt (kg)', 'Waist (cm)']] = df2[['kcal', 'kcal Total', 'Wgt (kg)', 'Waist (cm)']].apply(pd.to_numeric, errors='coerce')
    df2['Date'] = pd.to_datetime(df2['Date'], format='%d.%m.%Y', dayfirst=True)
    df2['BMI'] = df2['Wgt (kg)'] / ((height / 100) ** 2) 


    return df1, df2

def body_values(df2):

    start_date = '2024-04-01'
    start_date = pd.to_datetime(start_date)
    
    df2_bv = df2
    df2_bv = df2_bv[df2_bv['Date'] >= start_date]


    # Apply moving average to smooth the data
    df2_bv.loc[:, 'Wgt (kg)'] = df2_bv['Wgt (kg)'].rolling(window=1).mean()
    df2_bv.loc[:, 'Waist (cm)'] = df2_bv['Waist (cm)'].rolling(window=1).mean()
    df2_bv.loc[:, 'BMI'] = df2_bv['BMI'].rolling(window=1).mean()

    # Create a figure and axis
    fig, ax = plt.subplots(4, 1, figsize=(10, 8))

    sns.lineplot(data=df2_bv, x='Date', y='Waist (cm)', color=line_color, ax=ax[0])
    ax[0].set_title('Waist Over Time', fontweight='bold', fontsize=12, loc='left')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('')
    ax[0].set_xticklabels([])
    ax[0].tick_params(axis='y', labelsize=8)

    sns.lineplot(data=df2_bv, x='Date', y='Wgt (kg)', color=line_color, ax=ax[1])
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
    df_melted = df2_bv.melt(id_vars=['Date'], value_vars=['kcal', 'kcal Total'], var_name='Type', value_name='Value')
    custom_palette = {'kcal': line_color, 'kcal Total': 'lightgrey'}
    
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
    df_corr_1 = df2[['Waist (cm)', 'Wgt (kg)']].dropna()
    df_corr_1['Waist (cm)_MA'] = df_corr_1['Waist (cm)'].rolling(window=7).mean()
    df_corr_1['Wgt (kg)_MA'] = df_corr_1['Wgt (kg)'].rolling(window=7).mean()
    df_corr_1 = df_corr_1.dropna()

    correlation_matrix = df_corr_1[['Waist (cm)_MA', 'Wgt (kg)_MA']].corr()
    print("Waist vs. Weight Correlation Matrix")
    print(correlation_matrix)

    sns.heatmap(correlation_matrix, annot=True, cmap=color_palette, center=0, ax=ax)
    ax.set_title("Waist vs. Weight Correlation Matrix")

def correlation_weight_vs_kcal(df2, ax) -> None:
    df_corr_2 = df2[['Wgt (kg)', 'kcal']].dropna()
    print("Data for correlation_weight_vs_kcal before moving average:\n", df_corr_2.head())  # Debug print

    df_corr_2['Wgt (kg)_MA'] = df_corr_2['Wgt (kg)'].rolling(window=7).mean()
    df_corr_2['kcal_MA'] = df_corr_2['kcal'].rolling(window=7).mean()
    df_corr_2 = df_corr_2.dropna()
    print("Data for correlation_weight_vs_kcal after moving average:\n", df_corr_2.head())  # Debug print

    correlation_matrix = df_corr_2[['Wgt (kg)_MA', 'kcal_MA']].corr()
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

def main():
    df1, df2 = load_data(file_path_1, file_path_2)
    if df1 is not None and df2 is not None:
        df1, df2 = data_preparation(df1, df2)
        
        
        body_values(df2)

       

        '''
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        correlation_waist_v_weight(df2, axes[0])
        correlation_weight_vs_kcal(df2, axes[1])
        plt.show()

        sets_view(df1)
       
        
        
        '''

if __name__ == "__main__":
    main()
