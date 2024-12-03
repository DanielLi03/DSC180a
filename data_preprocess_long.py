import pandas as pd
import transformers as t

# read the csv from Toxic Comment Classification Challenge on Kaggle
df = pd.read_csv('train.csv.zip', compression='zip')

# get useful columns
df = df[['comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

# relevant row extraction to balance and get the data
first_100 = df.head(150)
severe_toxic = df[df['severe_toxic'] > 0.5].head(50)
obscene = df[df['obscene'] > 0.5].head(50)
threat = df[df['threat'] > 0.5].head(50)
insult = df[df['insult'] > 0.5].head(50)
identity_hate = df[df['identity_hate'] > 0.5 ].head(50)
combined_df = pd.concat([first_100, severe_toxic, obscene, threat, insult, identity_hate]).drop_duplicates()

# categorize the data as 1 or 0 depneding on if the text was of that specific trait
for i in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    combined_df[i] = combined_df[i].apply(lambda x: round(x))
combined_df = combined_df.rename(columns={'toxic': 'toxicity', 'severe_toxic': 'severe_toxicity', 'obscene': 'obscene', 'threat': 'threat', 'insult': 'insult', 'identity_hate': 'identity_attack'})

# export data into csv file
combined_df.to_csv('cleaned_data_long.csv', index=False)