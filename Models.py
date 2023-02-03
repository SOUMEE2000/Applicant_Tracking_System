import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
import nltk
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@st.cache
def get_HF_embeddings(sentences):

  # Load model from HuggingFace Hub
  tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
  model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
  # Tokenize sentences
  encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
  # Compute token embeddings
  with torch.no_grad():
      model_output = model(**encoded_input)
  # Perform pooling. In this case, max pooling.
  embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

  print("Sentence embeddings:")
  print(embeddings)
  return embeddings


@st.cache
def get_doc2vec_embeddings(JD, text_resume):
    nltk.download("punkt")
    data = [JD]

    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
    #print (tagged_data)

    model = gensim.models.doc2vec.Doc2Vec(vector_size=512, min_count=3, epochs=80)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=80)
    JD_embeddings = np.transpose(model.docvecs['0'].reshape(-1,1))
    text_resume = word_tokenize(text_resume.lower())
    resume_embeddings = model.infer_vector(text_resume)
    resume_embeddings = np.transpose(resume_embeddings.reshape(-1,1))
    return (JD_embeddings, resume_embeddings)


def cosine(embeddings1, embeddings2):
  # get the match percentage
  matchPercentage = cosine_similarity(np.array(embeddings1), np.array(embeddings2))
  matchPercentage = np.round(matchPercentage, 4)*100 # round to two decimal
  print("Your resume matches about" + str(matchPercentage[0])+ "% of the job description.")
  return str(matchPercentage[0][0])
