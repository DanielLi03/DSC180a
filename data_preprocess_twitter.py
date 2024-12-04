import pandas as pd
import transformers as t

# read the labeled_data.csv from t-davidson/hate-speech-and-offensive-language
df = pd.read_csv('labeled_data.csv')

# retrieve the first 200 rows of each class and relevant columns
toxic = df[df['class'] != 2][['class', 'tweet']].head(200)
nontoxic = df[df['class'] == 2][['class', 'tweet']].head(200)
combined_df = pd.concat([toxic, nontoxic])

# clean up and adjust the values of the columns
combined_df['tweet'] = combined_df['tweet'].str.strip('!\' \"RT@')
combined_df['class'] = combined_df['class'].apply(lambda x: 1 if x <= 1 else 0)

# export data into csv file
combined_df.to_csv('cleaned_data_twitter.csv', index=False)