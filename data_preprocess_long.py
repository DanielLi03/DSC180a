import pandas as pd
import transformers as t

# read the csv from Toxic Comment Classification Challenge on Kaggle
df = pd.read_csv('test_public_expanded.csv.zip', compression='zip')

# get useful columns
df = df[['comment_text', 'toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']]

# relevant row extraction to balance and get the data
first_20 = df.head(100)
severe_toxicity = df[df['severe_toxicity'] > 0.5].head(50)
obscene = df[df['obscene'] > 0.5].head(50)
threat = df[df['threat'] > 0.5].head(50)
insult = df[df['insult'] > 0.5].head(50)
identity_attack = df[df['identity_attack'] > 0.5 ].head(50)
combined_df = pd.concat([first_20, severe_toxicity, obscene, threat, insult, identity_attack]).drop_duplicates()

# categorize the data as 1 or 0 depneding on if the text was of that specific trait
for i in ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']:
    combined_df[i] = combined_df[i].apply(lambda x: round(x))

# export data into csv file
combined_df.to_csv('cleaned_data_long.csv', index=False)