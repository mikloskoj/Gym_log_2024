import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path_1 = r'C:\Users\janmi\Documents\VS Code\Gym_log_2024\gym_log_Q1_2024 - workout data.csv'
file_path_2 = r'C:\Users\janmi\Documents\VS Code\Gym_log_2024\gym_log_Q1_2024 - bio_data.csv'
body_weight = 79
height = 181
sns.set_style("white")
border_color = 'lightgrey'
background_color = '#fdfcfc'
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

    start_date = '2024-04-01'
    start_date = pd.to_datetime(start_date)
    df2 = df2[df2['Date'] >= start_date]

    return df1, df2


def body_values(df2):
    # Apply moving average to smooth the data
    df2['Wgt (kg)'] = df2['Wgt (kg)'].rolling(window=7).mean()
    df2['Waist (cm)'] = df2['Waist (cm)'].rolling(window=7).mean()
    df2['BMI'] = df2['BMI'].rolling(window=7).mean()

    # Create a figure and axis
    fig, ax = plt.subplots(5, 1, figsize=(10, 8))

    ax[0].text(0.5, 0.5, 'The first plot', ha='center', va='center', fontsize=8)
    ax[0].set_axis_off()  # Hide the axis for the text subplot

    sns.lineplot(data=df2, x='Date', y='Waist (cm)', color='#fed976', ax=ax[1])
    ax[1].set_title('Waist Over Time', fontweight='bold', fontsize=12, loc='left')
    ax[1].set_xlabel('')
    ax[1].set_ylabel('')
    ax[1].set_xticklabels([])
    ax[1].tick_params(axis='y', labelsize=8)

    sns.lineplot(data=df2, x='Date', y='Wgt (kg)', color='#fed976', ax=ax[2])
    ax[2].set_title('Weight Over Time', fontweight='bold', fontsize=12, loc='left')
    ax[2].set_xlabel('')
    ax[2].set_ylabel('')
    ax[2].set_xticklabels([])
    ax[2].tick_params(axis='y', labelsize=8)

    sns.lineplot(data=df2, x='Date', y='BMI', color='#fed976', ax=ax[3])
    ax[3].set_title('BMI Over Time', fontweight='bold', fontsize=12, loc='left')
    ax[3].set_xlabel('')
    ax[3].set_ylabel('')
    ax[3].set_xticklabels([])
    ax[3].tick_params(axis='y', labelsize=8)

    # Melt the DataFrame for the barplot
    df_melted = df2.melt(id_vars=['Date'], value_vars=['kcal', 'kcal Total'], var_name='Type', value_name='Value')
    custom_palette = {'kcal': '#feb24c', 'kcal Total': 'lightgrey'}
    
    sns.barplot(data=df_melted, x='Date', y='Value', hue='Type', palette=custom_palette, ax=ax[4])
    ax[4].set_title('Caloric Intake Over Time', fontweight='bold', fontsize=12, loc='left')
    ax[4].set_xlabel('')
    ax[4].set_ylabel('')
    ax[4].set_xticklabels([])
    ax[4].tick_params(axis='y', labelsize=8)

    for a in ax[1:]:
        a.spines['top'].set_color(border_color)
        a.spines['right'].set_color(border_color)
        a.spines['bottom'].set_color(border_color)
        a.spines['left'].set_color(border_color)
        sns.despine(ax=a, top=False, bottom=False, left=False, right=False)
        a.set_facecolor(background_color)

    plt.tight_layout()
    plt.show()


def correlation_waist_v_weight(df2, ax) -> None:
    # Select only the relevant columns
    df2 = df2[['Waist (cm)', 'Wgt (kg)']]
    df2 = df2.dropna()

    # Calculate the moving average with a window of 3 periods
    df2['Waist (cm)_MA'] = df2['Waist (cm)'].rolling(window=7).mean()
    df2['Wgt (kg)_MA'] = df2['Wgt (kg)'].rolling(window=7).mean()

    # Drop the rows with NaN values resulting from the moving average calculation
    df2 = df2.dropna()

    # Calculating the correlation matrix for the moving averages
    correlation_matrix = df2[['Waist (cm)_MA', 'Wgt (kg)_MA']].corr()

    # Displaying the correlation matrix
    print("Correlation Matrix for Waist vs. Weight with Moving Averages:")
    print(correlation_matrix)

    # Creating a heatmap to visualize the correlation matrix
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Waist vs. Weight Correlation Matrix Heatmap with Moving Averages")

    plt.tight_layout()
    plt.show()

def correlation_weight_vs_kcal(df2, ax) -> None:
    # Select only the relevant columns
    df2 = df2[['Wgt (kg)', 'kcal']]
    df2 = df2.dropna()

    # Calculate the moving average with a window of 3 periods
    df2['Wgt (kg)_MA'] = df2['Wgt (kg)'].rolling(window=7).mean()
    df2['kcal_MA'] = df2['kcal'].rolling(window=7).mean()

    # Drop the rows with NaN values resulting from the moving average calculation
    df2 = df2.dropna()

    # Calculating the correlation matrix for the moving averages
    correlation_matrix = df2[['Wgt (kg)_MA', 'kcal_MA']].corr()

    # Displaying the correlation matrix
    print("Correlation Matrix for Weight vs. Kcal with Moving Averages:")
    print(correlation_matrix)

    # Creating a heatmap to visualize the correlation matrix
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Weight vs. Kcal Correlation Matrix Heatmap with Moving Averages")

    plt.tight_layout()
    plt.show()


def main():
    df1, df2 = load_data(file_path_1, file_path_2)
    if df1 is not None and df2 is not None:
        df1, df2 = data_preparation(df1, df2)
        body_values(df2)
        
        # Create a figure with two subplots for the correlation functions
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        correlation_waist_v_weight(df2, axes[0])
        correlation_weight_vs_kcal(df2, axes[1])
        plt.show()


if __name__ == "__main__":
    main()
