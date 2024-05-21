import time
import traceback

from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification


def load_model():
    start_time = time.time()
    try:
        # 尝试下载和加载不同的模型
        tokenizer = DistilBertTokenizer.from_pretrained('bert-base-uncased')
        model = TFDistilBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        end_time = time.time()
        print(f"Model loaded in {end_time - start_time:.2f} seconds")
    except Exception as e:
        end_time = time.time()
        print(f"Failed to load model. Time taken: {end_time - start_time:.2f} seconds")
        print(f"Error: {e}")
        traceback.print_exc()


# 执行加载模型的函数
load_model()
