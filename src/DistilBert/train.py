import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from transformers import EarlyStoppingCallback
from datasets import Dataset


def load_data(data_path):
    df = pd.read_csv(data_path, sep=',', header=0)
    df = df[['label', 'text']]
    return df


def visualize_data_distribution_matplotlib(v_data):
    print(v_data['label'].value_counts())
    print(v_data['label'].value_counts(normalize=True))
    v_data['label'].value_counts(ascending=True).plot.barh()
    plt.title('Class Distribution')
    plt.show()


def count_characters(text):
    return len(text)


def test_exceed_max_tokens(v_data):
    v_data['Characters Per Sentence'] = v_data['text'].apply(count_characters)
    v_data.boxplot(column='Characters Per Sentence', by='label', grid=False, showfliers=False)
    plt.suptitle("")
    plt.xlabel("")
    plt.title('')
    plt.show()


def load_dataset(data):
    data_labels = data['label'].tolist()
    data_text = data['text'].tolist()
    train_text, val_text, train_labels, val_labels = train_test_split(data_text, data_labels, test_size=0.1,
                                                                      random_state=0)
    train_text, test_text, train_labels, test_labels = train_test_split(train_text, train_labels, test_size=0.05,
                                                                        random_state=0)

    # Load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert/distilbert-base-multilingual-cased')

    # Tokenize the datasets
    train_encodings = tokenizer(train_text, truncation=True, padding=True)
    val_encodings = tokenizer(val_text, truncation=True, padding=True)

    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })

    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': val_labels
    })

    return train_dataset, val_dataset, tokenizer


def loading_pretrained_model(save_dir):
    tokenizer_fine_tuned = DistilBertTokenizer.from_pretrained(save_dir)
    model_fine_tuned = DistilBertForSequenceClassification.from_pretrained(save_dir)
    return tokenizer_fine_tuned, model_fine_tuned


def load_test_data(data_path, tokenizer):
    df = pd.read_csv(data_path, sep=',', header=0)
    df = df[['label', 'text']]
    data_labels = df['label'].tolist()
    data_text = df['text'].tolist()

    # Tokenize the datasets
    test_encodings = tokenizer(data_text, truncation=True, padding=True, max_length=512)

    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': data_labels
    })

    return test_dataset


if __name__ == '__main__':
    data = load_data('../../data/toxic_comment_data/toxic_comment_train.csv')
    visualize_data_distribution_matplotlib(data)
    test_exceed_max_tokens(data)
    train_dataset, val_dataset, tokenizer = load_dataset(data)

    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=2,  # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=100,  # number of warmup steps for learning rate scheduler
        weight_decay=1e-5,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        eval_strategy="steps",  # ensure evaluation and save strategy match
        save_strategy="steps",  # ensure evaluation and save strategy match
        eval_steps=200,  # number of update steps between two evaluations
        logging_steps=100,  # log & save weights each logging_steps
        save_steps=400,  # save checkpoint every save_steps
        save_total_limit=2,  # limit the total amount of checkpoints. Deletes the older checkpoints.
        load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
        metric_for_best_model="eval_loss",  # set the metric to use to compare models
        report_to="tensorboard"  # report metrics to TensorBoard
    )

    # training_args = TrainingArguments(
    #     output_dir='./results',  # output directory
    #     num_train_epochs=5,  # total number of training epochs
    #     per_device_train_batch_size=16,  # batch size per device during training
    #     per_device_eval_batch_size=32,  # batch size for evaluation
    #     warmup_steps=100,  # number of warmup steps for learning rate scheduler
    #     weight_decay=1e-5,  # strength of weight decay
    #     logging_dir='./logs',  # directory for storing logs
    #     eval_strategy="steps",  # ensure evaluation and save strategy match
    #     save_strategy="steps",  # ensure evaluation and save strategy match
    #     eval_steps=200,  # number of update steps between two evaluations
    #     logging_steps=100,  # log & save weights each logging_steps
    #     save_steps=400,  # save checkpoint every save_steps
    #     save_total_limit=4,  # limit the total amount of checkpoints. Deletes the older checkpoints.
    #     load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
    #     metric_for_best_model="eval_loss",  # set the metric to use to compare models
    #     report_to="tensorboard"  # report metrics to TensorBoard
    # )
    print("Training arguments loaded")

    model = DistilBertForSequenceClassification.from_pretrained('distilbert/distilbert-base-multilingual-cased', num_labels=2)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]  # early stopping callback
    )

    trainer.train()
    trainer.evaluate()

    save_directory = './model'
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    # Load test data
    test_data_path = '../../data/toxic_comment_data/toxic_comment_test.csv'
    test_dataset = load_test_data(test_data_path, tokenizer)

    # Predict on test data
    predictions = trainer.predict(test_dataset)

    # Process predictions
    preds = predictions.predictions.argmax(-1)

    # Print the predictions and corresponding labels
    print("Predictions:", preds)
    print("Labels:", test_dataset['labels'])

    # Optional: calculate evaluation metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    labels = test_dataset['labels']
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
