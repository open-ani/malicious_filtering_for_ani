import os

import jieba, jieba.analyse, jieba.posseg

from .config_util import load_config

"""
这个文件主要是对jieba分词包的一些封装，包括初始化jieba，加载停用词，过滤词性和停用词等。
"""


def init_jieba(config_path: str) -> None:
    """

    :param config_path: config 文件地址
    :return: None
    """
    # 导入一些单词库来源：
    data_configs = load_config(config_path)["data_configs"]
    dictionaries = os.path.join(data_configs["external_raw_data_dir"], "dictionaries")
    for root, directories, files in os.walk(dictionaries):
        for filename in files:
            filepath = os.path.join(root, filename)
            jieba.load_userdict(filepath)

    # 停用词合集
    stopwords = os.path.join(data_configs["external_raw_data_dir"], "stopwords/stopword.txt")
    jieba.analyse.set_stop_words(stopwords)


def get_stop_words(stop_file_path: str) -> frozenset:
    """

    :param stop_file_path: 停用词列表的地址
    :return: 停用词集合
    """
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)


def filter_word_type_and_stopwords(text: str, stop_file_path: str) -> str:
    """

    :param text: 待处理的文本
    :param stop_file_path: 停用词列表的地址
    :return: 空格分隔的处理后的词语
    """
    stop_keys = get_stop_words(stop_file_path)
    word_list = []
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd', 'j', 'i', 'y', 'z', 'e', 'df', 'ad', 'an', 'b', 'ns', 'nrt', 'ng',
           'nrfg', 'nt', 'q', 'r', 'vi', 'nr', 'x']  # 定义选取的词性
    seg = jieba.posseg.cut(text)  # 分词
    for generator in seg:
        if generator.word not in stop_keys and generator.flag in pos:  # 去停用词 + 词性筛选
            word_list.append(generator.word)
    return ' '.join(word_list)
