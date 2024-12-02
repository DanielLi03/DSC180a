import streamlit as st
import transformers as t
from detoxify import Detoxify
from profanity_check import predict

def without_gruardrails(text):
    #insert english text
    translation = t.pipeline("translation_xx_to_yy", model="Helsinki-NLP/opus-mt-en-fr")
    return translation(text)[0]['translation_text']

def detoxify_test(text):
    results_detoxify = Detoxify('original').predict(without_gruardrails(text))
    if (max(results_detoxify.values()) > 0.5):
        return True
    return False

def profanity_test(text):
    return predict([text]) == 1

def tests(text, translation):
    if detoxify_test(text):
        st.success("Translation blocked by detoxify")
    else:
        st.success(translation)
    
    if profanity_test(text):
        st.success("Translation blocked by profanity check")
    else:
        st.success(translation)

def main():
    st.title("Guardrails Implementation in LLMs")

    text_area = st.text_area("Enter your text that you want to translate!")

    if st.button("Translate"):
        if len(text_area) > 0:
            st.info(text_area)
            st.warning("Translation Response Without Guardrails")
            without_gruardrails_result = without_gruardrails(text_area)
            st.success(without_gruardrails_result)
            st.warning("Translation Response With Guardrails")
            tests(text_area, without_gruardrails_result)

if __name__ == '__main__': 
    main()

