import pandas as pd

# Define the dataset
data = {
    'weather': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
    'play': ['yes', 'yes', 'yes', 'yes', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no']
}

# Convert the dataset into a Pandas DataFrame
df = pd.DataFrame(data)

# Convert the dataset into frequency tables
weather_freq = df['weather'].value_counts()
play_freq = df['play'].value_counts()

# Generate a Likelihood table by finding the probabilities of given features
likelihood = pd.crosstab(df['weather'], df['play'], margins=True, normalize='index')

# Use the Bayes theorem to calculate the posterior probability
# For example, let's calculate the probability of playing given that the weather is sunny
weather = 'sunny'
play = 'yes'
prior = play_freq[play] / len(df)
likelihood_given_weather = likelihood.loc[weather, play]
likelihood_given_not_weather = likelihood.loc[~(df['weather'] == weather), play].mean()
posterior = (likelihood_given_weather * prior) / ((likelihood_given_weather * prior) + (likelihood_given_not_weather * (1-prior)))

print(f"The probability of playing given that the weather is {weather} is {posterior}")
