import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="white")

# Load an example dataset with long-form data
data = {
    'City': ['Říčany', 'Praha'],
    'Population': [30,150]

}

df = pd.DataFrame(data)

# Plot the responses for different events and regions
sns.barplot(x="City", 
            y="Population",
            hue="City",
            data=df)

plt.show()