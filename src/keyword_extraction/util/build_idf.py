import pickle

import jieba
from collections import defaultdict
import os
import math
import jieba.posseg
import jieba.analyse
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump, load


def init_jieba():
    # 导入我找到的一些词 数据来源：https://github.com/bigzhao/Keyword_Extraction
    jieba.load_userdict('../../../data/字典/明星.txt')
    jieba.load_userdict('../../../data/字典/实体名词.txt')
    jieba.load_userdict('../../../data/字典/歌手.txt')
    jieba.load_userdict('../../../data/字典/动漫.txt')
    jieba.load_userdict('../../../data/字典/电影.txt')
    jieba.load_userdict('../../../data/字典/电视剧.txt')
    jieba.load_userdict('../../../data/字典/流行歌.txt')
    jieba.load_userdict('../../../data/字典/创造101.txt')
    jieba.load_userdict('../../../data/字典/百度明星.txt')
    jieba.load_userdict('../../../data/字典/美食.txt')
    jieba.load_userdict('../../../data/字典/FIFA.txt')
    jieba.load_userdict('../../../data/字典/NBA.txt')
    jieba.load_userdict('../../../data/字典/网络流行新词.txt')
    jieba.load_userdict('../../../data/字典/显卡.txt')

    # 爬取漫漫看网站和百度热点上面的词条，人名，英文组织
    jieba.load_userdict('../../../data/字典/漫漫看_明星.txt')
    jieba.load_userdict('../../../data/字典/百度热点人物+手机+软件.txt')
    jieba.load_userdict('../../../data/字典/自定义词典.txt')
    jieba.load_userdict('../../../data/字典/person.txt')
    jieba.load_userdict('../../../data/字典/origin_zimu.txt')
    jieba.load_userdict('../../../data/字典/出现的作品名字.txt')
    jieba.load_userdict('../../../data/字典/val_keywords.txt')

    # 停用词合集
    jieba.analyse.set_stop_words('../../../data/stopword.txt')

    # 敏感词合集：
    jieba.load_userdict('../../../data/字典/脏话.txt')
    jieba.load_userdict('../../../data/字典/色情词库cleaned.txt')


def get_stop_words(stop_file_path):
    """load stop words """

    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)


def dataPrepos(text, stopkey):
    l = []
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd', 'j', 'i', 'y', 'z', 'e', 'df', 'ad', 'an', 'b', 'ns', 'nrt', 'ng',
           'nrfg', 'nt', 'q', 'r', 'vi']  # 定义选取的词性
    seg = jieba.posseg.cut(text)  # 分词
    for i in seg:
        if i.word not in stopkey and i.flag in pos:  # 去停用词 + 词性筛选
            l.append(i.word)
    return ' '.join(l)


# this is used by jieba to do keyword extraction
def build_idf_from_folder(corpus_dir, output_file):
    documents = os.listdir(corpus_dir)
    print(documents)
    preprocessed_documents = []
    stop_words = get_stop_words('../../../data/stopword.txt')

    # Process each document
    for doc_name in documents:
        doc_path = os.path.join(corpus_dir, doc_name)
        print(f"Processing: {doc_name}")
        with open(doc_path, 'r', encoding='utf-8') as file:
            documents = file.read().split('\n')

        # Step 2: Preprocess each document using Jieba for tokenization
        preprocessed_docs = [dataPrepos(doc, stop_words) for doc in documents]
        preprocessed_documents += preprocessed_docs

    vectorizer = CountVectorizer(
        min_df=2,  # Minimum frequency for a word to be included
        max_df=0.85,  # Maximum proportion of documents a word can appear in
        ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
    )
    doc_term_matrix = vectorizer.fit_transform(preprocessed_docs)

    # Step 4: Get the terms and calculate document frequencies
    terms = vectorizer.get_feature_names_out()
    doc_frequencies = np.sum(doc_term_matrix.toarray() > 0, axis=0)

    # Step 5: Calculate IDF values
    N = len(documents)
    idf_values = {terms[i]: math.log(N / doc_frequencies[i]) for i in range(len(terms))}

    # Step 6: Save the IDF values to a new text file with a space separator
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for term, idf in idf_values.items():
            f_out.write(f"{term} {idf}\n")


def load_idf_file(file_path):
    """Loads an IDF file into a dictionary where keys are terms and values are IDF values."""
    idf_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            term, idf_value = line.split()
            idf_dict[term] = float(idf_value)
    return idf_dict


def combine_idf_files(file1, file2, file3, output_file, weights=(0.25, 0.25, 0.5)):
    """Combines three IDF files and writes the result to a new file using a weighted average."""
    # Load IDF dictionaries from all three files
    idf_dict1 = load_idf_file(file1)
    idf_dict2 = load_idf_file(file2)
    idf_dict3 = load_idf_file(file3)

    # Create a new dictionary to hold the combined IDF values
    combined_idf = {}

    # Set of all terms present in any of the dictionaries
    all_terms = set(idf_dict1.keys()).union(set(idf_dict2.keys())).union(set(idf_dict3.keys()))

    for term in all_terms:
        idf1 = idf_dict1.get(term, 0)  # Default to 0 if term not in dictionary
        idf2 = idf_dict2.get(term, 0)
        idf3 = idf_dict3.get(term, 0)

        # Weighted average of IDF values
        combined_idf[term] = (idf1 * weights[0]) + (idf2 * weights[1]) + (idf3 * weights[2])

    # Save the combined IDF values to the output file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for term, idf_value in combined_idf.items():
            f_out.write(f"{term} {idf_value}\n")


def clean_idf_file(input_file, output_file):
    """Cleans the input IDF file by ensuring each line has a term and IDF value."""
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) == 2:  # Ensure there are exactly two parts
                term, idf_value = parts
                try:
                    float(idf_value)  # Check if IDF value is numerical
                    f_out.write(f"{term} {idf_value}\n")
                except ValueError:
                    continue  # Skip lines with non-numerical IDF values

# ----------------------------- scikit-learn version -----------------------------


# this is used by scikit-learn to do keyword extraction
def build_tfidf_from_folder(corpus_dir, output_file, vectorizer_file='../../../data/vectorizer.pkl'):
    """
    Computes and saves the TF-IDF matrix for documents in the specified folder.

    Args:
    - corpus_dir (str): The path to the directory containing text documents.
    - output_file (str): The path to the output file where the TF-IDF matrix should be saved.
    - stopkey (set): A set of stop words to exclude during preprocessing.

    Returns:
    - None
    """
    documents = os.listdir(corpus_dir)  # List all documents in the corpus directory
    print(documents)

    # List to store preprocessed documents
    preprocessed_documents = []
    stop_words = get_stop_words('../../../data/stopword.txt')

    # Process each document
    for doc_name in documents:
        doc_path = os.path.join(corpus_dir, doc_name)
        print(f"Processing: {doc_name}")
        with open(doc_path, 'r', encoding='utf-8') as file:
            documents = file.read().split('\n')

        # Step 2: Preprocess each document using Jieba for tokenization
        preprocessed_docs = [dataPrepos(doc, stop_words) for doc in documents]
        preprocessed_documents += preprocessed_docs

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        min_df=2,  # Minimum frequency for a word to be included
        max_df=0.85,  # Maximum proportion of documents a word can appear in
        ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
        smooth_idf=True  # Apply smoothing to IDF scores
    )
    tfidf_matrix = vectorizer.fit_transform(preprocessed_documents)
    feature_names = vectorizer.get_feature_names_out()
    print(f"Number of terms: {len(feature_names)}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Save the TF-IDF matrix to the specified file
    np.savetxt(output_file, tfidf_matrix.toarray(), delimiter=',')
    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)


if __name__ == '__main__':
    init_jieba()
    corpus_file = '../../../data/corpus/'
    output_file = '../../../data/idfs/my_idf_from_corpus_folder_2.txt'
    build_idf_from_folder(corpus_file, output_file)
    # build_tfidf_from_folder(corpus_file, output_file)

    # idf_file1 = "../../data/my_idf_from_corpus_cleaned.txt"
    # idf_file2 = "../../data/3rd_party_idf_cleaned.txt"
    # idf_file3 = "../../data/my_idf_from_toxic.txt"
    # output_file = "../../data/idfs/combined_idf.txt"
    # combine_idf_files(idf_file1, idf_file2, idf_file3, output_file)
