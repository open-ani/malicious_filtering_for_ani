import bz2
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors

from src.keyword_extraction.keyword_extractor import init_jieba

import pandas as pd

# Load the CSV file
df = pd.read_csv('../../data/toxic_comment_data/toxic_comment_train_processed.csv')

# Check for NaNs in the specified column
nan_rows = df[df['TEXT'].isna()]

# Display the line numbers (index + 1) and optionally other columns
for index, row in nan_rows.iterrows():
    print(f"Line Number: {index + 2}, Data: {row}")


# init_jieba()
# # Load the Word2Vec model
# model_path = '../../../data/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.bz2'
# wv_from_text = KeyedVectors.load_word2vec_format(model_path, binary=False, unicode_errors='ignore')
#
#
# def sentence_to_vector(sentence):
#     print(sentence)
#     words = sentence.split()
#     word_vectors = [wv_from_text[word] for word in words if word in wv_from_text]
#     if len(word_vectors) == 0:
#         return np.zeros(300)  # Assuming the vectors are 300-dimensional
#     return np.mean(word_vectors, axis=0)
#
#
# df = pd.read_csv('../../data/toxic_comment_data/toxic_comment_train_processed.csv')
# # Apply to your DataFrame
# df['VECTOR'] = df['TEXT'].apply(sentence_to_vector)
# # Save the modified DataFrame back to CSV
# df.to_csv('../../../data/toxic_comment_data/toxic_comment_train_processed_with_vectors.csv', index=False)
