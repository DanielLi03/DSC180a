import pandas as pd
import numpy as np
from detoxify import Detoxify
from profanity_check import predict

# read cleaned subset dataset taken from Toxic Comment Classification Challenge on Kaggle
df = pd.read_csv('data/cleaned_data_wiki.csv')

# function to run the basic python guardrail
def py_profanity_check(text):
    return predict(text)

# function to run the detoxify guardrail
def detoxify_check(text):
    return Detoxify('original').predict(text)

# get the sample text
text_data = df['comment_text']

# get the toxicity verdict for each comment
toxicity = df['toxicity']
severe_toxicity = df['severe_toxicity']
obscene = df['obscene']
threat = df['threat']
insult = df['insult']
identity_attack = df['identity_attack']

# profanity check tests and results
profanity_check_results = py_profanity_check(text_data)
profanity_check_accuracy = sum(i == j for i, j in zip(profanity_check_results, toxicity)) / len(profanity_check_results)
profanity_check_TN = sum(((i == j) and (j == 0)) for i, j in zip(profanity_check_results, toxicity)) / (len(toxicity) - sum(toxicity))
profanity_check_TP = sum(((i == j) and (j == 1)) for i, j in zip(profanity_check_results, toxicity)) / (sum(toxicity))

# add profanity_check results to df for convenience
df['profanity_check_results'] = profanity_check_results

# function to check the accuracy of the profanity_cehck guardrail within each type of toxicity
def profanity_check_toxic_type_result(col):
    if len(df[df[col] == 1]) > 0:
        return sum(i == j for i, j in zip(df[df[col] == 1]['profanity_check_results'], df[df[col] == 1]['toxicity'])) / len(df[df[col] == 1]['toxicity'])
    else:
        return 'Not enough data'
    
# compile profanity_check_results
profanity_check_severe_toxicity_accuracy = profanity_check_toxic_type_result('severe_toxicity')
profanity_check_obscene_accuracy = profanity_check_toxic_type_result('obscene')
profanity_check_threat_accuracy = profanity_check_toxic_type_result('threat')
profanity_check_insult_accuracy = profanity_check_toxic_type_result('insult')
profanity_check_identity_attack_accuracy = profanity_check_toxic_type_result('identity_attack')

# print profanity_check result as a dictionary
print({'Overall Accuracy': profanity_check_accuracy, 'True Positive Rate': profanity_check_TP, 'True Negative Rate': profanity_check_TN, \
'Severe Toxicity Accuracy': profanity_check_severe_toxicity_accuracy, 'Obscene Accuracy': profanity_check_obscene_accuracy, \
'Threat Accuracy': profanity_check_threat_accuracy, 'Insult Accuracy': profanity_check_insult_accuracy, \
'Identity Attack Accuracy': profanity_check_identity_attack_accuracy})

# run detoxify test and organize outputs
detoxify_results = detoxify_check(list(text_data))
df_results = pd.DataFrame(detoxify_results)
for i in ['toxicity']:
    df_results[i] = df_results[i].apply(lambda x: round(x))
df['detoxify_results'] = df_results['toxicity']

# retrieve results for detoxify tests
detoxify_accuracy = sum(i == j for i, j in zip(df['detoxify_results'], toxicity)) / len(toxicity)
detoxify_TN = sum(((i == j) and (j == 0)) for i, j in zip(df['detoxify_results'], toxicity)) / (len(toxicity) - sum(toxicity))
detoxify_TP = sum(((i == j) and (j == 1)) for i, j in zip(df['detoxify_results'], toxicity)) / (sum(toxicity))

# function to check the accuracy of the detoxify guardrail within each type of toxicity
def detoxify_toxic_type_result(col):
    if len(df[df[col] == 1]) > 0:
        return sum(i == j for i, j in zip(df[df[col] == 1]['detoxify_results'], df[df[col] == 1]['toxicity'])) / len(df[df[col] == 1]['toxicity'])
    else:
        return 'Not enough data'

# compile detoxify results
detoxify_severe_toxicity_accuracy = detoxify_toxic_type_result('severe_toxicity')
detoxify_obscene_accuracy = detoxify_toxic_type_result('obscene')
detoxify_threat_accuracy = detoxify_toxic_type_result('threat')
detoxify_insult_accuracy = detoxify_toxic_type_result('insult')
detoxify_identity_attack_accuracy = detoxify_toxic_type_result('identity_attack')

# output detoxify results
print({'Overall Accuracy': detoxify_accuracy, 'True Positive Rate': detoxify_TP, 'True Negative Rate': detoxify_TN, \
'Severe Toxicity Accuracy': detoxify_severe_toxicity_accuracy, 'Obscene Accuracy': detoxify_obscene_accuracy, \
'Threat Accuracy': detoxify_threat_accuracy, 'Insult Accuracy': detoxify_insult_accuracy, \
'Identity Attack Accuracy': detoxify_identity_attack_accuracy})

# compile results if we ran both guardrails in succession
combined_results = np.array(df_results['toxicity']) + np.array(profanity_check_results)
combined_results = [1 if i > 0 else i for i in combined_results]

# calculate combined results
combined_accuracy = sum(i == j for i, j in zip(combined_results, toxicity)) / len(combined_results)
combined_TN = sum(((i == j) and (j == 0)) for i, j in zip(combined_results, toxicity)) / (len(toxicity) - sum(toxicity))
combined_TP = sum(((i == j) and (j == 1)) for i, j in zip(combined_results, toxicity)) / (sum(toxicity))

# add combined results to df
df['combined_results'] = combined_results

# function to check the accuracy of the combined guardrail within each type of toxicity
def combined_toxic_type_result(col):
    if len(df[df[col] == 1]) > 0:
        return sum(i == j for i, j in zip(df[df[col] == 1]['combined_results'], df[df[col] == 1]['toxicity'])) / len(df[df[col] == 1]['toxicity'])
    else:
        return 'Not enough data'

# compile combined results
combined_severe_toxicity_accuracy = combined_toxic_type_result('severe_toxicity')
combined_obscene_accuracy = combined_toxic_type_result('obscene')
combined_threat_accuracy = combined_toxic_type_result('threat')
combined_insult_accuracy = combined_toxic_type_result('insult')
combined_identity_attack_accuracy = combined_toxic_type_result('identity_attack')

# output combined results
print({'Overall Accuracy': combined_accuracy, 'True Positive Rate': combined_TP, 'True Negative Rate': combined_TN, \
'Severe Toxicity Accuracy': combined_severe_toxicity_accuracy, 'Obscene Accuracy': combined_obscene_accuracy, \
'Threat Accuracy': combined_threat_accuracy, 'Insult Accuracy': combined_insult_accuracy, \
'Identity Attack Accuracy': combined_identity_attack_accuracy})

