from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd
import json

# Load the fine-tuned model and tokenizer
save_directory = './model'
tokenizer = DistilBertTokenizer.from_pretrained(save_directory)
model = DistilBertForSequenceClassification.from_pretrained(save_directory)


def preprocess(texts, tokenizer):
    # Tokenize the input texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
    return encodings


def load_json_data(file):
    with open(file, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    comments = json_data['comments']
    text = []
    for c in comments:
        text.append(c['m'])
    print(text[:10])
    return text


# texts = load_json_data('dandanplaytest.json')
texts = ['这个视频真的很好看', '这个视频真的很垃圾', '懂不懂啊傻逼', '不是主角脑子感觉有点问题', "虎头蛇尾的动漫", '什么东西啊，无语了']
inputs = preprocess(texts, tokenizer)


def predict(inputs, model):
    # Put the model in evaluation mode
    model.eval()

    # Move inputs to the same device as the model (if using GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs


outputs = predict(inputs, model)

# calculate result and confidence and save into a excel file

# Get the predicted class and confidence

predicted_class = torch.argmax(outputs.logits, dim=1)
confidence = torch.nn.functional.softmax(outputs.logits, dim=1).max(dim=1).values

# Save the results to an Excel file

df = pd.DataFrame({
    'Comment': texts,
    'Predicted Class': predicted_class.cpu().numpy(),
    'Confidence': confidence.cpu().numpy()
})

print(df)

# if predicted_class == 1, it is toxic comment, turn line to red when saving in excel

# df.to_excel('predictions.xlsx', index=True)
