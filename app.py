import streamlit as st
from transformers import pipeline

st.header("NER SYSTEM")
text = st.text_input("Enter your text")
if st.button("Get Entity") :
    ner= pipeline("ner", model="xlm-roberta-large-finetuned-conll03-english", aggregation_strategy="simple")
    result = ner(text)
    for entity in result:
        st.write(f"Entity: {entity['word']} | Label: {entity['entity_group']}")
