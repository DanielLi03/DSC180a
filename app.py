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
    results_detoxify = Detoxify('original').predict(without_gruardrails(text))
    # return true if one of the toxicity categories is over 0.5
    if (max(results_detoxify.values()) > 0.5):
        return True
    return False

# function to run the python profanity check package
def profanity_test(text):
    return predict([text]) == 1

# function to return approriate output based on detoxify and profanity check results
def tests(detoxify, profanity, translation):
    # detoxify output
    st.warning("Translation Response With Detoxify Guardrail")
    if detoxify:
        st.success("Translation blocked by Detoxify")
    else:
        st.success(translation)
    
    # profanity check output
    st.warning("Translation Response With Profanity Check Guardrail")
    if profanity:
        st.success("Translation blocked by Profanity Check")
    else:
        st.success(translation)
    
    # final output
    st.warning("Final Output")
    if detoxify or profanity:
        st.success("Translation was blocked by Guardrail")
    else:
        st.success(translation)

# main code to run the app
def main():
    # title and instructions
    st.title("Guardrails Implementation in Translation LLMs")

    text_area = st.text_area("Enter your text that you want to translate!")

    # main logic of the app
    if st.button("Translate"):
        if len(text_area) > 0:
            st.info("Input: " + text_area)
            st.warning("Translation Response Without Guardrails")
            without_gruardrails_result = without_gruardrails(text_area)
            st.success(without_gruardrails_result)
            detoxify_result = detoxify_test(text_area)
            profanity_result = profanity_test(text_area)
            tests(detoxify_result, profanity_result, without_gruardrails_result)


if __name__ == '__main__': 
    main()
