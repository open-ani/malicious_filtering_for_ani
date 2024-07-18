import pickle
import time

import jieba
from jieba import analyse
import numpy as np

from src.keyword_extraction.util.build_idf import dataPrepos, get_stop_words


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
    # 敏感词合集：
    jieba.load_userdict('../../data/字典/脏话.txt')
    jieba.load_userdict('../../data/字典/色情词库cleaned.txt')

    # 停用词合集
    jieba.analyse.set_stop_words('../../data/stopword.txt')

    # tf-idf语料：https://github.com/codemayq/chinese-chatbot-corpus
    jieba.analyse.set_idf_path('../../data/idfs/my_idf_from_corpus_folder_2_cleaned.txt')


def get_tokens(text):
    tokens = list(jieba.cut(text))
    return tokens


def get_keywords(text, k):
    # 用Jieba的TF-IDF分析关键词
    keywords = jieba.analyse.extract_tags(text, topK=k)
    return keywords


# def extract_keywords(new_text, vectorizer_file):
#     """
#     Extracts keywords from a given text using a previously built TF-IDF model.
#
#     Args:
#     - new_text (str): The new text document from which to extract keywords.
#     - vectorizer_file (str): The path to the file where the fitted TfidfVectorizer model is stored.
#     - stopkey (set): A set of stop words to exclude during preprocessing.
#
#     Returns:
#     - List[str]: A list of top keywords from the text.
#     """
#     stopkey = get_stop_words('../../data/stopword.txt')
#     # Preprocess the text
#     preprocessed_text = dataPrepos(new_text, stopkey)
#
#     # Load the TF-IDF vectorizer
#     with open(vectorizer_file, 'rb') as f:
#         vectorizer = pickle.load(f)
#
#     # Transform the new text into a TF-IDF vector
#     tfidf_vector = vectorizer.transform([preprocessed_text])
#
#     # Extract keywords
#     row = tfidf_vector.toarray()[0]
#
#     # Get the feature names
#     feature_names = vectorizer.get_feature_names_out()
#
#     # Get the indices of terms sorted by their TF-IDF scores
#     top_indices = np.argsort(row)[::-1]  # Sort in descending order
#
#     # Return the top 10 terms
#     top_keywords = [feature_names[idx] for idx in top_indices[:10]]
#
#     return top_keywords


if __name__ == '__main__':
    init_jieba()
    lines = []
    with open('../../data/test.txt', 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()  # This will read the file and split by newlines
    start = time.time()
    for l in lines:
        print("Keyword: " + str(get_keywords(l, 5)))
    end = time.time()
    print("Time: " + str(end - start))
    # seg = jieba.posseg.cut("不错＋1，后面那个台湾人回山东拜年的，我不拿它当小品，而是当一个温情情景剧，说实话也还可以，但是蔡明那个节目真的是"
    #                        "让我恶心到了，他妈的不能好好说话吗？非要这么做作，当全国观众都是需要大人捏嗓子逗的婴儿？")
    # for s in seg:
    #     print(s.word + "_" + s.flag + "\n")
