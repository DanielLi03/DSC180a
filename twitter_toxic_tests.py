import pandas as pd
import numpy as np
from detoxify import Detoxify
from profanity_check import predict

# read cleaned subset dataset taken from t-davidson/hate-speech-and-offensive-language
df = pd.read_csv('data/cleaned_data_twitter.csv')

# function to run the basic python guardrail
def py_profanity_check(text):
    return predict(text)

# function to run the detoxify guardrail
def detoxify_check(text):
    return Detoxify('original').predict(text)

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

# compile results if we ran both guardrails in succession
combined_results = np.array(df_results['toxicity']) + np.array(profanity_check_results)
combined_results = [1 if i > 0 else i for i in combined_results]

# calculate final results
final_accuracy = sum(i == j for i, j in zip(combined_results, toxicity)) / len(combined_results)
final_TN = sum(((i == j) and (j == 0)) for i, j in zip(combined_results, toxicity)) / (len(toxicity) - sum(toxicity))
final_TP = sum(((i == j) and (j == 1)) for i, j in zip(combined_results, toxicity)) / (sum(toxicity))

# print final results as a dictionary
print({'Overall Accuracy': final_accuracy, 'True Positive Rate': final_TP, 'True Negative Rate': final_TN})