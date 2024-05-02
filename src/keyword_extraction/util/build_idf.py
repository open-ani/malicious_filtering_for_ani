import jieba
from collections import defaultdict
import os
import math
import jieba.posseg
import jieba.analyse
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump, load


def init_jieba():
    # 导入我找到的一些词 数据来源：https://github.com/bigzhao/Keyword_Extraction
    jieba.load_userdict('../../data/字典/明星.txt')
    jieba.load_userdict('../../data/字典/实体名词.txt')
    jieba.load_userdict('../../data/字典/歌手.txt')
    jieba.load_userdict('../../data/字典/动漫.txt')
    jieba.load_userdict('../../data/字典/电影.txt')
    jieba.load_userdict('../../data/字典/电视剧.txt')
    jieba.load_userdict('../../data/字典/流行歌.txt')
    jieba.load_userdict('../../data/字典/创造101.txt')
    jieba.load_userdict('../../data/字典/百度明星.txt')
    jieba.load_userdict('../../data/字典/美食.txt')
    jieba.load_userdict('../../data/字典/FIFA.txt')
    jieba.load_userdict('../../data/字典/NBA.txt')
    jieba.load_userdict('../../data/字典/网络流行新词.txt')
    jieba.load_userdict('../../data/字典/显卡.txt')

    # 爬取漫漫看网站和百度热点上面的词条，人名，英文组织
    jieba.load_userdict('../../data/字典/漫漫看_明星.txt')
    jieba.load_userdict('../../data/字典/百度热点人物+手机+软件.txt')
    jieba.load_userdict('../../data/字典/自定义词典.txt')
    jieba.load_userdict('../../data/字典/person.txt')
    jieba.load_userdict('../../data/字典/origin_zimu.txt')
    jieba.load_userdict('../../data/字典/出现的作品名字.txt')
    jieba.load_userdict('../../data/字典/val_keywords.txt')

    # 停用词合集
    jieba.analyse.set_stop_words('../../../data/stopword.txt')

    # 敏感词合集：
    jieba.load_userdict('../../data/字典/脏话.txt')
    jieba.load_userdict('../../data/字典/色情词库cleaned.txt')


def clean_dictionary(input_file, output_file):
    """Removes numerical values from each line in the input file,
    leaving only the words, and writes the cleaned data to the output file."""

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            word = line.split()[0]
            f_out.write(f"{word}\n")


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


# def build_tf_idf_model(corpus_file, output_file=None):
#     stop_words = get_stop_words("../../data/stopword.txt")
#     with open(corpus_file, 'r', encoding='utf-8') as file:
#         documents = file.read().split('\n')
#     print("Pre-Processing")
#     preprocessed_docs = [dataPrepos(doc, stop_words) for doc in documents]
#     cv = CountVectorizer(max_df=0.9)
#     print("Fitting.")
#     word_count_vector = cv.fit_transform(preprocessed_docs)
#     print(word_count_vector.shape)
#     cv_file = '../../data/cv.joblib'
#     dump(cv, cv_file)
#     transformer_file = '../../data/tfidf_transformer.joblib'
#     if os.path.exists(transformer_file):
#         tfidf_transformer = load("../../data/tfidf_transformer.joblib")
#     else:
#         tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
#         tfidf_transformer.fit(word_count_vector)
#         dump(tfidf_transformer, transformer_file)
#
#     tfidf_matrix = tfidf_transformer.transform(word_count_vector)
#     print(tfidf_matrix.shape)


def build_idf_from_file(corpus_file, output_file):
    # Step 1: Load the corpus into a list of documents
    with open(corpus_file, 'r', encoding='utf-8') as file:
        documents = file.read().split('\n')

    stop_words = get_stop_words('../../../data/stopword.txt')

    # Step 2: Preprocess each document using Jieba for tokenization
    preprocessed_docs = [dataPrepos(doc, stop_words) for doc in documents]

    # Step 3: Create the document-term matrix using CountVectorizer
    vectorizer = CountVectorizer()
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


# def build_idf_from_folder(corpus_dir, output_file):
#     """Builds an IDF file from a corpus of text documents."""
#     term_df = defaultdict(int)  # Dictionary to store term document frequencies
#     documents = os.listdir(corpus_dir)  # List all documents in the corpus directory
#     total_docs = len(documents)
#
#     # Process each document
#     for doc_name in documents:
#         print(doc_name)
#         doc_path = os.path.join(corpus_dir, doc_name)
#
#         with open(doc_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#
#         # Tokenize the content
#         terms = set(jieba.cut(content))  # Use a set to avoid duplicate terms
#
#         # Update document frequency for each unique term
#         for term in terms:
#             term_df[term] += 1
#
#     # Write the IDF values to the output file
#     with open(output_file, 'w', encoding='utf-8') as f_out:
#         for term, df in term_df.items():
#             idf_value = math.log((total_docs / (df + 1)) + 1)
#             f_out.write(f"{term} {idf_value}\n")


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


if __name__ == '__main__':
    init_jieba()
    # corpus_dir = '../data/clean_chat_corpus/'  # Directory containing text documents
    # output_file = '../data/my_idf_from_corpus.txt'  # File to store IDF values
    # build_idf_file(corpus_dir, output_file)

    # input_file = "../../data/3rd_party_idf.txt"
    # output_file = "../../data/3rd_party_idf_cleaned.txt"
    #
    # clean_idf_file(input_file, output_file)
    corpus_file = '../../../data/toxic_comment_data/toxic_comment_train_text.txt'
    output_file = '../../../data/idfs/my_idf_from_toxic.txt'
    build_idf_from_file(corpus_file, output_file)

    # build_tf_idf_model(corpus_file)

    # idf_file1 = "../../data/my_idf_from_corpus_cleaned.txt"
    # idf_file2 = "../../data/3rd_party_idf_cleaned.txt"
    # idf_file3 = "../../data/my_idf_from_toxic.txt"
    # output_file = "../../data/idfs/combined_idf.txt"
    # combine_idf_files(idf_file1, idf_file2, idf_file3, output_file)
