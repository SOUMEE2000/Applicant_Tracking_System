import streamlit as st
import pdfplumber
from Models import get_HF_embeddings, cosine, get_dco2vec_embeddings


def extract_data(feed):
    data = ""
    with pdfplumber.open(feed) as pdf:
        pages = pdf.pages
        for page in pages:
            data += page.extract_text()
    return data # build more code to return a dataframe


def compare(uploaded_file, JD, flag = 'HuggingFace-BERT'):

    if flag == 'HuggingFace-BERT':
        JD_embeddings = None
        resume_embeddings = None

        if uploaded_file is not None:
            df = extract_data(uploaded_file)
            resume_embeddings = get_HF_embeddings(df)
        if JD is not None:
            JD_embeddings = get_HF_embeddings(JD)
        if JD_embeddings is not None and resume_embeddings is not None:
            cos = cosine(resume_embeddings, JD_embeddings)
            st.write("Score is: ", cos)

    else:
        if uploaded_file is not None:
            df = extract_data(uploaded_file)

        JD_embeddings, resume_embeddings = get_doc2vec_embeddings(JD, df)
        cos = cosine(resume_embeddings, JD_embeddings)
        st.write("Cosine similarity is: ", cos)
