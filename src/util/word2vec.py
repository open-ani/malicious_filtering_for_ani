import bz2

# Open the bz2 file in text read mode
with bz2.open('../data/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.bz2', 'rt') as f:  # 'rt' for reading as text
    # Print the first 4 lines
    for i in range(4):
        line = f.readline()  # Read one line at a time
        if not line:  # Break if there's no more content
            break
        print(line.strip())  # Print the line, removing any leading or trailing whitespace
