from jieba import analyse

# 引入TF-IDF关键词抽取接口
tfidf = analyse.extract_tags

# 原始文本
text = "train_docs_keywords.txt"

# 基于TF-IDF算法进行关键词抽取
keywords = tfidf(text)
print ("keywords by tfidf:")
# 输出抽取出的关键词
for keyword in keywords:
    print(keyword + "/",)