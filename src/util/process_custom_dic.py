def clean_dictionary(input_file, output_file):
    """Removes numerical values from each line in the input file,
    leaving only the words, and writes the cleaned data to the output file."""

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            # Split the line by whitespace to get the word
            word = line.split()[0]  # Get the first element (the word)

            # Write the word to the output file
            f_out.write(f"{word}\n")


# Example usage
input_file = '../data/字典/色情词库.txt'  # Input file containing words and numbers
output_file = '../data/字典/色情词库cleaned.txt'  # Output file to store only the words

clean_dictionary(input_file, output_file)
