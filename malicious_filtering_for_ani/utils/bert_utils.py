import pandas as pd
import matplotlib.pyplot as plt
import logging


def load_training_data_df(data_path: str, text_column: str, label_column: str) -> pd.DataFrame:
    """
    载入训练数据
    :param label_column: 标签数据的列名
    :param text_column: 文本数据的列名
    :param data_path: 数据文件的路径
    :return: DataFrame
    """
    df = pd.read_csv(data_path, sep=',', header=0)
    df = df[[label_column, text_column]]
    df['Characters Per Sentence'] = df[text_column].apply(len)
    if max(df['Characters Per Sentence']) > 512:
        logging.warning('Some text exceeds the maximum token limit of 512')
    return df


def visualize_data_distribution_matplotlib(v_data, column_name: str) -> None:
    """
    Visualize the distribution of the data using matplotlib
    :param v_data: the data to visualize, should have a 'label' column
    :return: None
    """
    print(v_data[column_name].value_counts())
    print(v_data[column_name].value_counts(normalize=True))
    v_data[column_name].value_counts(ascending=True).plot.barh()
    plt.title(f'{column_name} Column Class Distribution')
    plt.show()
