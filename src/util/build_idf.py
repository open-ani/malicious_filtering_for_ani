import os
import jieba
import math
from collections import defaultdict

from jieba import analyse

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

# 敏感词合集：
jieba.load_userdict('../data/字典/脏话.txt')
jieba.load_userdict('../data/字典/色情词库cleaned.txt')

def build_idf_file(corpus_dir, output_file):
    """Builds an IDF file from a corpus of text documents."""
    term_df = defaultdict(int)  # Dictionary to store term document frequencies
    documents = os.listdir(corpus_dir)  # List all documents in the corpus directory
    total_docs = len(documents)

    # Process each document
    for doc_name in documents:
        print(doc_name)
        doc_path = os.path.join(corpus_dir, doc_name)

        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Tokenize the content
        terms = set(jieba.cut(content))  # Use a set to avoid duplicate terms

        # Update document frequency for each unique term
        for term in terms:
            term_df[term] += 1

    # Write the IDF values to the output file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for term, df in term_df.items():
            idf_value = math.log((total_docs / (df + 1)) + 1)
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



corpus_dir = '../data/clean_chat_corpus/'  # Directory containing text documents
output_file = '../data/my_idf1.txt'  # File to store IDF values
build_idf_file(corpus_dir, output_file)

input_file = "../data/my_idf1.txt"
output_file = "../data/cleaned_idf.txt"

clean_idf_file(input_file, output_file)

