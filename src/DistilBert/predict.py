from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
save_directory = './model'
tokenizer = DistilBertTokenizer.from_pretrained(save_directory)
model = DistilBertForSequenceClassification.from_pretrained(save_directory)


def preprocess(texts, tokenizer):
    # Tokenize the input texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
    return encodings


texts = ["我真的服了我靠，你是真的狗", "我草泥马，你是真的狗"]
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

print("Predictions: " + str(outputs))


def postprocess(outputs):
    # Get the predicted class labels
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions


predictions = postprocess(outputs)

# Map the prediction indices to their corresponding labels
label_map = {0: 'Negative', 1: 'Positive'}  # Adjust this based on your dataset's labels
predicted_labels = [label_map[pred.item()] for pred in predictions]

print(predicted_labels)