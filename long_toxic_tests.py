import pandas as pd
from detoxify import Detoxify
from profanity_check import predict

# read cleaned subset dataset taken from Toxic Comment Classification Challenge on Kaggle
df = pd.read_csv('cleaned_data_long.csv')

# function to run the basic python guardrail
def py_profanity_check(text):
    return predict(text)

# function to run the detoxify guardrail
def detoxify_check(text):
    return Detoxify('unbiased').predict(text)

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

# print profanity_check result as a dictionary
print({'Overall Accuracy': profanity_check_accuracy, 'True Positive Rate': profanity_check_TP, 'True Negative Rate': profanity_check_TN})

# run detoxify test and organize outputs
detoxify_results = detoxify_check(list(text_data))
df_results = pd.DataFrame(detoxify_results)
for i in ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']:
    df_results[i] = df_results[i].apply(lambda x: round(x))

# retrieve results for detoxify tests
detoxify_accuracy = sum(i == j for i, j in zip(df_results['toxicity'], toxicity)) / len(toxicity)
detoxify_TN = sum(((i == j) and (j == 0)) for i, j in zip(df_results['toxicity'], toxicity)) / (len(toxicity) - sum(toxicity))
detoxify_TP = sum(((i == j) and (j == 1)) for i, j in zip(df_results['toxicity'], toxicity)) / (sum(toxicity))

# function to check the accuracy of the detoxify guardrail within each type of toxicity
def toxic_type_result(col):
    if len(df[df[col] == 1]) > 0:
        return sum(i == j for i, j in zip(df_results[df_results[col] == 1]['toxicity'], df[df[col] == 1]['toxicity'])) / len(df[df[col] == 1]['toxicity'])
    else:
        return 'Not enough data'

# compile results
detoxify_severe_toxicity_accuracy = toxic_type_result('severe_toxicity')
detoxify_obscene_accuracy = toxic_type_result('obscene')
detoxify_threat_accuracy = toxic_type_result('threat')
detoxify_insult_accuracy = toxic_type_result('insult')
detoxify_identity_attack_accuracy = toxic_type_result('identity_attack')

# output results
print({'Overall Accuracy': detoxify_accuracy, 'True Positive Rate': detoxify_TP, 'True Negative Rate': detoxify_TN, \
'Severe Toxicity Accuracy': detoxify_severe_toxicity_accuracy, 'Obscene Accuracy': detoxify_obscene_accuracy, \
'Threat Accuracy': detoxify_threat_accuracy, 'Insult Accuracy': detoxify_insult_accuracy, \
'Identity Attack Accuracy': detoxify_identity_attack_accuracy})
