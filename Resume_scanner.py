import streamlit as st
import pdfplumber
from Models import get_HF_embeddings, cosine, get_doc2vec_embeddings


def extract_data(feed):
    data = ""
    with pdfplumber.open(feed) as pdf:
        pages = pdf.pages
        for page in pages:
            data += page.extract_text()
    return data # build more code to return a dataframe


def compare(uploaded_files, JD, flag = 'HuggingFace-BERT'):

    if flag == 'HuggingFace-BERT':
        JD_embeddings = None
        resume_embeddings = []

        if JD is not None:
            JD_embeddings = get_HF_embeddings(JD)
        if uploaded_files is not None:
            for i in uploaded_files:
                df = extract_data(i)
                resume_embeddings.append(get_HF_embeddings(df))
        if JD_embeddings is not None and resume_embeddings is not None:
            cos = cosine(resume_embeddings, JD_embeddings)
            #st.write("Score is: ", cos)

    else:
        df = []
        if uploaded_files is not None:
            for i in uploaded_files:
                data = extract_data(i)
                df.append(data)

        JD_embeddings, resume_embeddings = get_doc2vec_embeddings(JD, df)
        if JD_embeddings is not None and resume_embeddings is not None:
            cos = cosine(resume_embeddings, JD_embeddings)
        #st.write("Cosine similarity is: ", cos)
    return cos
