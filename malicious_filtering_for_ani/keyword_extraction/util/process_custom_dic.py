import pandas as pd
import re


def extract_text_column(input_file, output_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file, delimiter=',')

    # Extract the 'TEXT' column
    text_column = df["review"]

    # Save the column to a new file
    text_column.to_csv(output_file, index=False, header=False)


def clean_text_line(line):
    """
    Cleans a single line by removing spaces between words and retaining spaces between numbers and words.
    """
    # Step 1: Remove spaces between words
    text_without_word_spaces = re.sub(r'([^\d\s])\s+([^\d\s])', r'\1\2', line)

    # Step 2: Ensure spaces are kept between numbers and words
    text_with_number_word_spaces = re.sub(r'(\d)\s+([^\d])', r'\1 \2', text_without_word_spaces)
    text_with_number_word_spaces = re.sub(r'([^\d])\s+(\d)', r'\1 \2', text_with_number_word_spaces)

    return text_with_number_word_spaces


def process_text_file(input_file, output_file):
    """
    Reads text from the input file line by line, applies cleaning transformations,
    and saves the modified text to the output file.

    Args:
    - input_file (str): Path to the input text file.
    - output_file (str): Path to the output text file.
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        # Read and process each line
        for line in f_in:
            cleaned_line = clean_text_line(line)

            # Write the cleaned line to the output file, including the newline character
            f_out.write(cleaned_line)

    print(f"Processed text saved to {output_file}")

if __name__ == '__main__':
    # Replace these paths with the actual paths for the input and output files
    input_csv = "../../../data/idfs/my_idf_from_corpus_folder_2.txt"
    output_txt = "../../../data/idfs/my_idf_from_corpus_folder_2_cleaned.txt"

    process_text_file(input_csv, output_txt)

    # Example usage
    # input_file = '../data/字典/色情词库.txt'  # Input file containing words and numbers
    # output_file = '../data/字典/色情词库cleaned.txt'  # Output file to store only the words
    #
    # clean_dictionary(input_file, output_file)
