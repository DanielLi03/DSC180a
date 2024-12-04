# DSC180a

## Project Overview
My Quarter 1 Guardrail Replication Project focused on implementing a profanity and toxicity guardrail on basic English to French translation app. The LLM I used was Helsinki-NLP/opus-mt-en-fr on hugging face, which is a powerful translation LLM that is part of the Opus family, hosted on hugging face. The particular model I used translates English language to French. My intended idea of the translation app is so that you can use it to translate tweets from English to French, while censoring toxic or hateful tweets during the translation process. This way, I can promote a positive social media environment across multi-lingual barriers, even in an app like Twitter. As such, I focused on toxicity guardrails that were trained on twitter datasets, which classified each tweet as toxic or not. I used two guardrails called Detoxify and Profanity_check. Detoxify is ML model trained on the Jigsaw Unintended Bias in Toxicity Classification Dataset and the Toxic Comment Classification Challenge, both on Kaggle. Profanity_check is an ML model trained on a curated dataset hosted on github at t-davidson/hate-speech-and-offensive-language, as well as the Toxic Comment Classification Challenge on Kaggle. My project shows the effectiveness of both guardrails, as well as their synergy together to provide an efficient and function guardrail for a translation app

## Files Overview

- The images folder contains all the images used in this README file
- app.py contains all the code for the translation app. We run this to see the actual app.
- cleaned_data_twitter.csv contains the twitter dataset used to test the effectiveness of the guardrails
- cleaned_data_wiki.csv contains the wikipedia dataset used to test the effectiveness of the guardrails
- data_preprocess_twitter.py contains the code used to generate the cleaned_data_twitter.csv
- data_preprocess_wiki.py contains the code used to generate the cleaned_data_wiki.csv
- environment.yml contains all the information of our anaconda environment. Use this to create the anaconda environment
- requirements.txt contains a list of all the packages used in our project (environment.yml uses requirements.txt to download all the pip packages)
- twitter_toxic_test.py contains the code used to test the effectiveness of our guardrails on our cleaned_data_twitter.csv dataset
- wiki_toxic_test.py contains the code used to test the effectiveness of our guardrails on our cleaned_data_wiki.csv dataset

## How to Install and Setup
To run this project, we need to install both Python and Anaconda onto our device, so please make sure you have that installed. Once you have those two installed, you can proceed to the next step.

First we need to clone the respository. Open up a terminal, navigate to the desired folder and clone the respository as follows:

```
git clone https://github.com/DanielLi03/DSC180a.git
```

Next we need to create a conda environment. Open up an anaconda terminal, navigate to the cloned repository and run the following command

```
conda env create -f environment.yml
```

By now, all our packages should be installed in our anaconda environment so we should be ready to run the code now (see How to Run Tests).

## How to Run Tests
To run both the Twitter and Wikipedia toxic tests of our guardrails, activate the anaconda environment that you've created in an anaconda terminal. We do this by typing in
```
conda activate DSC180a-Daniel-Li 
```

Navigate to the github repo on your local device in the conda terminal, and run the python file as follows:
```
python twitter_toxic_tests.py
```

Note: these tests will take several minutes to run, and might download some ML model if it's your first time running it. There will also be warnings, but they shouldn't affect the ability to run the code at all. Ignoring the warnings, the expected output after running both python in the terminal should be three dictionaries containing various metrics on the accuracy of the Profanity Check, Detoxify, and Combined Guardrails respectively. If the code has run correctly, we should expect to see the following result in the terminal:

![twitter toxic test results](/images/twitter_test_results.png)

For the wiki_toxic_test.py, simply run in your conda environment.
```
python wiki_toxic_tests.py
```

Again, this test might take several minutes to run. Ignoring the warnings, the expected output after running both python in the terminal should be three dictionaries containing various metrics on the accuracy of the Profanity Check, Detoxify, and Combined Guardrails respectively. If the code has run correctly, we should expect to see the following result in the terminal:

![wikipedia toxic test results](/images/wiki_test_results.png)

## How to Run App.py
To run the actual app, again, we need to activate the anaconda environment (you can use the same terminal you used to run your tests) and navigate to the github repo. In our conda terminal, run

```
streamlit run app.py
```

If that doesn't work, run 

```
python -m streamlit run app.py
```

Now in your browser, you should see a simple translation app with a singular textbox as below:

![naked translation app](/images/app.png)

If you type whatever english phrase into the text box, and click translate, the app should translate your english phrase into french, and output different results based on the toxicivity and nature of the english phrase you entered. Here is an example 

![clean translation example](/images/clean_app_example.png)

Here is a toxic example:

![toxic translation example](/images/toxic_app_example.png)

Here is an example that gets flagged by only Detoxify:

![detoxify translation example](/images/detoxify.png)

Here is an example that get flagged by only Profanity Check:

![Profanity Check translation example](/images/profanity.png)

Note that the first time you run this app, it will take a couple minutes becuase the LLM translation model is being installed. However, after the first instance of running the app, it should take much less time (maybe a minute or two at most, depending on the length of your prompt).
