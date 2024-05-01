import jieba
from jieba import analyse


class keyword_extractor:

    def __init__(self, k=5):
        self.k = k
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

        # 停用词合集
        jieba.analyse.set_stop_words('../data/stopword.txt')

        # tf-idf语料：https://github.com/codemayq/chinese-chatbot-corpus
        jieba.analyse.set_idf_path("../data/cleaned_idf.txt")

        # 敏感词合集：
        jieba.load_userdict('../data/字典/脏话.txt')
        jieba.load_userdict('../data/字典/色情词库cleaned.txt')

    def get_tokens(self, text):
        tokens = list(jieba.cut(text))
        return tokens

    def get_keywords(self, text):
        # 用Jieba的TF-IDF分析关键词
        keywords = jieba.analyse.extract_tags(text, topK=5)
        return keywords
