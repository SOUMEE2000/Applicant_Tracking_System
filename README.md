## Instructions

**Requirements:**
```
pip install streamlit
pip install transformers
pip install pytorch
pip install pdfplumber
```
**Run**: ``` streamlit run application.py```

## Interface
<img src = "https://github.com/SOUMEE2000/Resume_Scanner/blob/main/Demo/Interface.png" height=500>

## Intuition:
1. Get context-aware BERT Embeddings for Resume and Job Description, using the [Hugging Face](https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens) Library
2. Get their cosine similarity
