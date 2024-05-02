import time

import jieba
from jieba import analyse, posseg
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump, load

from src.util.build_idf import get_stop_words, dataPrepos


def init_jieba():
    # 导入我找到的一些词 数据来源：https://github.com/bigzhao/Keyword_Extraction
    jieba.load_userdict('../data/字典/明星.txt')
    jieba.load_userdict('../data/字典/实体名词.txt')
    jieba.load_userdict('../data/字典/歌手.txt')
    jieba.load_userdict('../data/字典/动漫.txt')
    jieba.load_userdict('../data/字典/电影.txt')
    jieba.load_userdict('../data/字典/电视剧.txt')
    jieba.load_userdict('../data/字典/流行歌.txt')
    jieba.load_userdict('../data/字典/创造101.txt')
    jieba.load_userdict('../data/字典/百度明星.txt')
    jieba.load_userdict('../data/字典/美食.txt')
    jieba.load_userdict('../data/字典/FIFA.txt')
    jieba.load_userdict('../data/字典/NBA.txt')
    jieba.load_userdict('../data/字典/网络流行新词.txt')
    jieba.load_userdict('../data/字典/显卡.txt')

    # 爬取漫漫看网站和百度热点上面的词条，人名，英文组织
    jieba.load_userdict('../data/字典/漫漫看_明星.txt')
    jieba.load_userdict('../data/字典/百度热点人物+手机+软件.txt')
    jieba.load_userdict('../data/字典/自定义词典.txt')
    jieba.load_userdict('../data/字典/person.txt')
    jieba.load_userdict('../data/字典/origin_zimu.txt')
    jieba.load_userdict('../data/字典/出现的作品名字.txt')
    jieba.load_userdict('../data/字典/val_keywords.txt')
    # 敏感词合集：
    jieba.load_userdict('../data/字典/脏话.txt')
    jieba.load_userdict('../data/字典/色情词库cleaned.txt')

    # 停用词合集
    jieba.analyse.set_stop_words('../data/stopword.txt')

    # tf-idf语料：https://github.com/codemayq/chinese-chatbot-corpus
    jieba.analyse.set_idf_path('../data/idfs/my_idf_from_toxic.txt')


def get_tokens(text):
    tokens = list(jieba.cut(text))
    return tokens


def get_keywords(text, k):
    # 用Jieba的TF-IDF分析关键词
    keywords = jieba.analyse.extract_tags(text, topK=k)
    return keywords


# def get_keywords_from_model(text, k):
#     tfidf_transformer = load("../data/tfidf_transformer.joblib")
#     vectorizer = load("../data/cv.joblib")
#     stop_words = get_stop_words("../data/stopword.txt")
#     text = dataPrepos(text, stop_words)
#     # Convert text into a list
#     texts = [text]
#     word_count_vector = vectorizer.transform(texts)
#     tfidf_matrix = tfidf_transformer.transform(word_count_vector)
#     terms = vectorizer.get_feature_names_out()
#
#     tfidf_scores = tfidf_matrix.toarray()[0]  # Convert sparse matrix to an array
#
#     keywords = sorted(zip(terms, tfidf_scores), key=lambda x: x[1], reverse=True)
#
#     for term, score in keywords[:k]:
#         print(f"Keyword: {term}, Score: {score}")


if __name__ == '__main__':
    init_jieba()
    text = ("也是，想想物流爆仓等半个月和打砸抢烧丧尸围城一般冒着被黑人胖大妈一屁股坐死的风险半夜2点去门店排队，我宁愿等着，慢就慢吧，命比较重要")
    start = time.time()
    print(text)
    print("tokens" + str(get_tokens(text)))
    print("Keyword: " + str(get_keywords(text, 5)))
    end = time.time()
    print("Time: " + str(end - start))
    # seg = jieba.posseg.cut("不错＋1，后面那个台湾人回山东拜年的，我不拿它当小品，而是当一个温情情景剧，说实话也还可以，但是蔡明那个节目真的是"
    #                        "让我恶心到了，他妈的不能好好说话吗？非要这么做作，当全国观众都是需要大人捏嗓子逗的婴儿？")
    # for s in seg:
    #     print(s.word + "_" + s.flag + "\n")
