import jieba.analyse
from src.keyword_extraction.keyword_extractor import init_jieba

if __name__ == '__main__':

    init_jieba()
    with open('../../data/toxic_comment_data/toxic_comment_train_text.txt', 'r', encoding='utf-8') as file:
        documents = file.read().split('\n')[:10]
    for doc in documents:
        keywords = jieba.analyse.textrank(doc, topK=5, allowPOS=('n','nz','v','vd','vn','l','a','d'))

        print(doc)
        print(keywords)