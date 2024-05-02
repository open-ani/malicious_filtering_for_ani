import pandas as pd


def extract_text_column(input_file, output_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Extract the 'TEXT' column
    text_column = df["TEXT"]

    # Save the column to a new file
    text_column.to_csv(output_file, index=False, header=False)


if __name__ == '__main__':
    # Replace these paths with the actual paths for the input and output files
    input_csv = "../../data/toxic_comment_train.csv"
    output_txt = "../../data/toxic_comment_train_text.txt"

    extract_text_column(input_csv, output_txt)

    # Example usage
    # input_file = '../data/字典/色情词库.txt'  # Input file containing words and numbers
    # output_file = '../data/字典/色情词库cleaned.txt'  # Output file to store only the words
    #
    # clean_dictionary(input_file, output_file)
