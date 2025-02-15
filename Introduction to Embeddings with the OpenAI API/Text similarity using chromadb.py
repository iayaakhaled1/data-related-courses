!pip install chromadb

import chromadb
from chromadb.utils import embedding_functions
chroma_client = chromadb.Client()


def create_embedding(input_text):
  default_ef = embedding_functions.DefaultEmbeddingFunction()
  val = default_ef(input_text)
  # print(val) 
  return val


llm_types = [
    {"type": "Conversational AI", "features": "Handles multi-turn conversations, generates human-like responses."},
    {"type": "Text Summarization", "features": "Produces concise summaries of long documents, captures key points."},
    {"type": "Sentiment Analysis", "features": "Determines the sentiment (positive, negative, neutral) in text."},
    {"type": "Question Answering", "features": "Provides accurate answers to questions based on context."},
    {"type": "Text Classification", "features": "Categorizes text into predefined classes or labels."},
    {"type": "Named Entity Recognition", "features": "Identifies and classifies named entities in text (e.g., names, dates)."},
    {"type": "Language Translation", "features": "Translates text from one language to another with high accuracy."},
    {"type": "Text Generation", "features": "Generates coherent and contextually relevant text based on input."},
    {"type": "Text Embedding", "features": "Creates numerical representations of text for semantic understanding."},
    {"type": "Text Similarity", "features": "Measures the similarity between two pieces of text based on their embeddings."}
]

def create_embedding(input_text):
  default_ef = embedding_functions.DefaultEmbeddingFunction()
  features_embd = default_ef(input_text)
  return features_embd


for item in llm_types:
  item['Embedding'] = create_embedding(item['features'])


text_to_match_Similarity = ['human text']
text_to_match_Similarity_embedding = create_embedding(text_to_match_Similarity)
print(text_to_match_Similarity_embedding)
type(text_to_match_Similarity_embedding[0])

from scipy.spatial import distance
import numpy as np



distance_list = []

# Calculate cosine distance
for item in llm_types:
    cosine_dist = distance.cosine(item['Embedding'][0], text_to_match_Similarity_embedding[0])
    distance_list.append({'distance': cosine_dist, 'type': item['type']})

for result in distance_list:
    print(f"Type: {result['type']}, Cosine Distance: {result['distance']}")

  

min_val =   min([item['distance'] for item in distance_list])
for item in distance_list:
  if item['distance']== min_val:
    print(item['type'])
