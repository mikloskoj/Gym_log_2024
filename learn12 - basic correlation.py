import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Explanation of correlation coefficients
# 1: Perfect positive correlation. As one variable increases, the other variable increases proportionally.
# 0: No correlation. The variables do not affect each other.
# -1: Perfect negative correlation. As one variable increases, the other variable decreases proportionally.

sns.set_theme(style="white")

def main() -> None:
    df = pd.read_csv(r'C:\Users\janmi\Documents\VS Code\Gym_log_2024\gym_log_Q1_2024 - bio_data.csv', sep=';', encoding='latin1')

    # Replace comma with dot for weight column
    df['Wgt (kg)'] = df['Wgt (kg)'].str.replace(',', '.').astype(float)
    df['Waist (cm)'] = df['Waist (cm)'].str.replace(',', '.').astype(float)
    df['kcal'] = df['kcal'].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    correlation_waist_v_weight(df, axes[0])
    correlation_weight_vs_kcal(df, axes[1])

    plt.tight_layout()
    plt.show()

def correlation_waist_v_weight(df, ax) -> None:
    # Select only the relevant columns
    df = df[['Waist (cm)', 'Wgt (kg)']]
    df = df.dropna()

    # Calculate the moving average with a window of 3 periods
    df['Waist (cm)_MA'] = df['Waist (cm)'].rolling(window=7).mean()
    df['Wgt (kg)_MA'] = df['Wgt (kg)'].rolling(window=7).mean()

    # Drop the rows with NaN values resulting from the moving average calculation
    df = df.dropna()

    # Calculating the correlation matrix for the moving averages
    correlation_matrix = df[['Waist (cm)_MA', 'Wgt (kg)_MA']].corr()

    # Displaying the correlation matrix
    print("Correlation Matrix for Waist vs. Weight with Moving Averages:")
    print(correlation_matrix)

    # Creating a heatmap to visualize the correlation matrix
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Waist vs. Weight Correlation Matrix Heatmap with Moving Averages")

def correlation_weight_vs_kcal(df, ax) -> None:
    # Select only the relevant columns
    df = df[['Wgt (kg)', 'kcal']]
    df = df.dropna()

    # Calculate the moving average with a window of 3 periods
    df['Wgt (kg)_MA'] = df['Wgt (kg)'].rolling(window=7).mean()
    df['kcal_MA'] = df['kcal'].rolling(window=7).mean()

    # Drop the rows with NaN values resulting from the moving average calculation
    df = df.dropna()

    # Calculating the correlation matrix for the moving averages
    correlation_matrix = df[['Wgt (kg)_MA', 'kcal_MA']].corr()

    # Displaying the correlation matrix
    print("Correlation Matrix for Weight vs. Kcal with Moving Averages:")
    print(correlation_matrix)

    # Creating a heatmap to visualize the correlation matrix
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Weight vs. Kcal Correlation Matrix Heatmap with Moving Averages")

main()
