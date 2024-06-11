import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_path = r'C:\Users\janmi\Documents\VS Code\Gym_log_2024\gym_log_Q1_2024 - bio_data.csv'
body_weight = 79
height = 181
start_date = '2024-04-01'
start_date = pd.to_datetime(start_date)
border_color = 'lightgrey' 
background_color = '#fdfcfc'


def moving_average(series, window_size):
    '''Calculating moving_average'''
    return series.rolling(window=window_size, min_periods=1).mean()


def body_values(df) -> None:

    # Apply moving average to smooth the data
    df['Wgt (kg)'] = moving_average(df['Wgt (kg)'], window_size=2)
    df['Waist (cm)'] = moving_average(df['Waist (cm)'], window_size=2)
    df['BMI'] = moving_average(df['BMI'], window_size=2)

    sns.set_style("white")

    # Create a figure and axis
    fig, ax = plt.subplots(5, 1, figsize=(10, 8))

    ax[0].text(0.5, 0.5, 'The first plot', ha='center', va='center', fontsize=8)
    ax[0].set_axis_off()  # Hide the axis for the text subplot
    

    sns.lineplot(data=df, 
                 x='Date', 
                 y='Waist (cm)', 
                 color='#90AACB',
                 ax=ax[1])
    
    
    ax[1].set_title('Waist Over Time', fontweight='bold', fontsize='9', loc='left')
    ax[1].set_xlabel('')
    ax[1].set_ylabel('')
    ax[1].set_xticklabels([])
    

    sns.lineplot(data=df, 
                 x='Date', 
                 y='Wgt (kg)', 
                 color='#90AACB',
                 ax=ax[2])
    

    ax[2].set_title('Weight Over Time', fontweight='bold', fontsize='9', loc='left')
    ax[2].set_xlabel('')
    ax[2].set_ylabel('')
    ax[2].set_xticklabels([])


    sns.lineplot(data=df, 
                 x='Date', 
                 y='BMI', 
                 color='#90AACB',
                 ax=ax[3])
    
    ax[3].set_title('BMI Over Time', fontweight='bold', fontsize='9', loc='left')
    ax[3].set_xlabel('')
    ax[3].set_ylabel('')
    ax[3].set_xticklabels([])


    # Melt the DataFrame for the barplot
    df_melted = df.melt(id_vars=['Date'], value_vars=['kcal', 'kcal Total'], var_name='Type', value_name='Value')
    custom_palette = {'kcal': '#FFB085', 'kcal Total': 'lightgrey'}
    
    sns.barplot(data=df_melted, 
                x='Date', 
                y='Value', 
                hue='Type', 
                palette=custom_palette,
                ax=ax[4])
    
    ax[4].set_title('Caloric Intake Over Time', fontweight='bold', fontsize='9', loc='left')
    ax[4].set_xlabel('')
    ax[4].set_ylabel('')
    ax[4].set_xticklabels([])
    


    
    for a in ax[1:]:
        a.spines['top'].set_color(border_color)
        a.spines['right'].set_color(border_color)
        a.spines['bottom'].set_color(border_color)
        a.spines['left'].set_color(border_color)
        sns.despine(ax=a, top=False, bottom=False, left=False, right=False) 
        a.set_facecolor(background_color)

    
    plt.tight_layout()
    plt.show()

def main() -> None:
    try:
        df = pd.read_csv(file_path, encoding='latin1', sep=';')

        df[['kcal', 'kcal Total', 'Wgt (kg)', 'Waist (cm)']] = df[['kcal', 'kcal Total', 'Wgt (kg)', 'Waist (cm)']].astype(str).apply(lambda x: x.str.replace(',', '.'))
        df[['kcal', 'kcal Total', 'Wgt (kg)', 'Waist (cm)']] = df[['kcal', 'kcal Total', 'Wgt (kg)', 'Waist (cm)']].apply(pd.to_numeric, errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y',dayfirst=True)
        df['BMI'] = df['Wgt (kg)'] / ((height / 100) ** 2)
        df = df[df['Date'] >= start_date]
        
        
    


    except FileNotFoundError as e:
        print(f'File not found: {e}')
    else:
        body_values(df)

main()

