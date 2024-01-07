
import streamlit as st
from Models import get_HF_embeddings, cosine, get_doc2vec_embeddings

def compare(resume_texts, JD_text, flag='HuggingFace-BERT'):
    JD_embeddings = None
    resume_embeddings = []

    if flag == 'HuggingFace-BERT':
        if JD_text is not None:
            JD_embeddings = get_HF_embeddings(JD_text)
        for resume_text in resume_texts:
            resume_embeddings.append(get_HF_embeddings(resume_text))

        if JD_embeddings is not None and resume_embeddings is not None:
            cos_scores = cosine(resume_embeddings, JD_embeddings)
            return cos_scores

    # Add logic for other flags like 'Doc2Vec' if necessary
    else:
        # Handle other cases
        pass