import jieba

sent = '中文分词是文本处理不可或缺的一步!'

seg_list = jieba.cut(sent)
print('默认精确模式：', '/ '.join(seg_list))

