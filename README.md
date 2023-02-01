# Resume Screening App
This app is built for employers looking for candidates against a particular job description. This app looks into outputing a x% percent similarity score given the resume of the candidate and a job description.

## Intuition:
1. Get [context-aware BERT Embeddings](https://towardsdatascience.com/nlp-extract-contextualized-word-embeddings-from-bert-keras-tf-67ef29f60a7b) for Resume and Job Description, using the [Hugging Face](https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens) Library
2. Get their [cosine similarity](https://developers.google.com/machine-learning/clustering/similarity/measuring-similarity)

## Interface
<img src = "https://github.com/SOUMEE2000/Resume_Scanner/blob/main/Demo/Interface.png" height=500>

## Instructions

**Requirements:**
```
pip install streamlit
pip install transformers
pip install pytorch
pip install pdfplumber
```
**Run**: ``` streamlit run application.py```

