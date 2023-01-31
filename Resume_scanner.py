from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(sentences):
  sentences = JD
  
  # Load model from HuggingFace Hub
  tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
  model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

  # Tokenize sentences
  encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)

  # Compute token embeddings
  with torch.no_grad():
      model_output = model(**encoded_input)

  # Perform pooling. In this case, max pooling.
  JD_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

  print("Sentence embeddings:")
  print(JD_embeddings)


def cosine_similarity(embeddings1, embeddings2):
  # get the match percentage
  matchPercentage = cosine_similarity(np.array(sentence_embeddings), np.array(JD_embeddings))
  #mathPercentage = cosine_similarity(predictions_resume)
  matchPercentage = np.round(matchPercentage, 4)*100 # round to two decimal
  print("Your resume matches about" + str(matchPercentage)+ "% of the job description.")

  if __name__ == "main":
    #do something
