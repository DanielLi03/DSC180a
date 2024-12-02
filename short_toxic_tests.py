import pandas as pd
from detoxify import Detoxify
from profanity_check import predict

# read cleaned subset dataset taken from t-davidson/hate-speech-and-offensive-language
df = pd.read_csv('cleaned_data_short.csv')

# function to run the basic python guardrail
def py_profanity_check(text):
    return predict(text)

# function to run the detoxify guardrail
def detoxify_check(text):
    return Detoxify('unbiased').predict(text)

# get text data and toxicity score
text_data = df['tweet']
toxicity = df['class']

# profanity check tests and results
profanity_check_results = py_profanity_check(text_data)
profanity_check_accuracy = sum(i == j for i, j in zip(profanity_check_results, toxicity)) / len(profanity_check_results)
profanity_check_TN = sum(((i == j) and (j == 0)) for i, j in zip(profanity_check_results, toxicity)) / (len(profanity_check_results) - sum(profanity_check_results))
profanity_check_TP = sum(((i == j) and (j == 1)) for i, j in zip(profanity_check_results, toxicity)) / (sum(profanity_check_results))

# print profanity_check result as a dictionary
print({'Overall Accuracy': profanity_check_accuracy, 'True Positive Rate': profanity_check_TP, 'True Negative Rate': profanity_check_TN})

# detoxify tests
detoxify_results = detoxify_check(list(text_data))
df_results = pd.DataFrame(detoxify_results)[['toxicity']]
df_results['toxicity'] = df_results['toxicity'].apply(lambda x: round(x))

# calculate detoxify results
detoxify_accuracy = sum(i == j for i, j in zip(df_results['toxicity'], toxicity)) / len(toxicity)
detoxify_TN = sum(((i == j) and (j == 0)) for i, j in zip(df_results['toxicity'], toxicity)) / (len(df_results['toxicity']) - sum(df_results['toxicity']))
detoxify_TP = sum(((i == j) and (j == 1)) for i, j in zip(df_results['toxicity'], toxicity)) / (sum(df_results['toxicity']))

# print detoxify result as a dicitionary
print({'Overall Accuracy': detoxify_accuracy, 'True Positive Rate': detoxify_TP, 'True Negative Rate': detoxify_TN})