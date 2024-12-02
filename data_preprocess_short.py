import pandas as pd
import transformers as t

# read the csv from t-davidson/hate-speech-and-offensive-language
df = pd.read_csv('labeled_data.csv')

# retrieve the first 250 rows and relevant columns
df = df[['class', 'tweet']].head(250)

# clean up and adjust the values of the columns
df['tweet'] = df['tweet'].str.strip('!\' \"RT@')
df['class'] = 2 - df['class']

# export data into csv file
df.to_csv('cleaned_data_short.csv', index=False)