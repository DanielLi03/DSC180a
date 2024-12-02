# DSC180a

## Project Overview
My Quarter 1 Guardrail Replication Project focused on implementing a profanity and toxicity guardrail on basic English to French translation app. The LLM I used was Helsinki-NLP/opus-mt-en-fr on hugging face, which is a powerful translation LLM that is part of the Opus family, hosted on hugging face. The particular model I used translates English language to French. My intended idea of the translation app is so that you can use it to translate tweets from English to French, while censoring toxic or hateful tweets during the translation process. This way, I can promote a positive social media environment across multi-lingual barriers, even in an app like Twitter. As such, I focused on toxicity guardrails that were trained on twitter datasets, which classified each tweet as toxic or not. I used two guardrails called Detoxify and Profanity_check. Detoxify is ML model trained on the Jigsaw Unintended Bias in Toxicity Classification Dataset and the Toxic Comment Classification Challenge, both on Kaggle. Profanity_check is an ML model trained on a curated dataset hosted on github at t-davidson/hate-speech-and-offensive-language, as well as the Toxic Comment Classification Challenge on Kaggle.

## How to Install

 - First we need to clone the repository 

## How to Run Tests

## How to Run App.py
