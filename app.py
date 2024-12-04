import streamlit as st
import transformers as t
from detoxify import Detoxify
from profanity_check import predict

# translation function without guadrail
def without_gruardrails(text):
    #insert english text into pipeline
    translation = t.pipeline("translation_xx_to_yy", model="Helsinki-NLP/opus-mt-en-fr")
    return translation(text)[0]['translation_text']

# function to run detoxify guardrail
def detoxify_test(text):
    results_detoxify = Detoxify('unbiased').predict(text)
    # return true if one of the toxicity categories is over 0.5
    if (max(results_detoxify.values()) > 0.5):
        return True
    return False

# function to run the python profanity check package
def profanity_test(text):
    return predict([text]) == 1

# function to return approriate output based on detoxify and profanity check results
def tests(detoxify, profanity, translation):
    # check if input passes the guardrails and output appropriate response
    st.warning("Translation Response with Guardrails")
    if detoxify and profanity:
        st.success("Translation was blocked by Detoxify and Profanity Check Guardrail")
    elif detoxify:
        st.success("Translation was blocked by Detoxify Guardrail")
    elif profanity:
        st.success("Translation was blocked by Profanity Check Guardrail")
    else:
        st.success(translation)

# main code to run the app
def main():
    # title and instructions
    st.title("Profanity Check and Detoxify Guardrails Implementation in English to French Translation LLMs")

    text_area = st.text_area("Enter your text that you want to translate!")

    # main logic of the app
    if st.button("Translate"):
        if len(text_area) > 0:
            # instructions
            st.info("Input: " + text_area)
            # translated response without guardrails
            st.warning("Translation Response Without Guardrails")
            without_gruardrails_result = without_gruardrails(text_area)
            st.success(without_gruardrails_result)
            # translated response with guardrails
            detoxify_result = detoxify_test(text_area)
            profanity_result = profanity_test(text_area)
            tests(detoxify_result, profanity_result, without_gruardrails_result)


if __name__ == '__main__': 
    main()

