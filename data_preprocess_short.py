import pandas as pd
import transformers as t

# read the labeled_data.csv from t-davidson/hate-speech-and-offensive-language
df = pd.read_csv('labeled_data.csv')

# retrieve the first 400 rows and relevant columns
df = df[['class', 'tweet']].head(400)

# clean up and adjust the values of the columns
df['tweet'] = df['tweet'].str.strip('!\' \"RT@')
df['class'] = 2 - df['class']
df['class'] = 2 - df['class']
df['class'] = df['class'].apply(lambda x: 1 if x >= 1 else 0)

# export data into csv file
df.to_csv('cleaned_data_short.csv', index=False)