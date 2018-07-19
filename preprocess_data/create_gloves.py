# Create Glove dictionary for the words in the dataset to use pretrained glove embeddings

import argparse
from nltk.tokenize import TweetTokenizer
import io
import sys
sys.path.insert(0, './..')
from utils.file_handlers import pickle_dump

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Creating GLOVE dictionary.. Please first download http://nlp.stanford.edu/data/glove.42B.300d.zip')

    parser.add_argument("-data_path", type=str, default='../data/train2.txt' , help="Path to dataset")
    parser.add_argument("-glove_in", type=str, default="../data/glove.42B.300d.txt", help="Name of the stanford glove file")
    parser.add_argument("-glove_out", type=str, default="../glove_dict.pkl", help="Name of the output glove file")

    args = parser.parse_args()

    tokenizer = TweetTokenizer(preserve_case=False)
    
    print("Loading glove...")
    with io.open(args.glove_in, 'r', encoding="utf-8") as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    # define the tokens that are to be replaced in the trainset (those starting with 'u' are in unicode format)
    tokens_to_replace = [u"\u2019s", u"\u2019ve", u"\u2019t", u"\u2019", u"\u201c", u"\u201d", "(", ")", "[", "]", "{", "}",
                    ".", ",", "?", "*", "!", ";", ":", "_"]
    
    print("Mapping glove...")
    glove_dict = {}
    not_in_dict = {}
    path = args.data_path # train data in txt format
    total_words = 0
    with io.open(path, 'r', encoding="utf8") as f:
        for line in f:
            words = line.split()
            total_words += len(words)-1
            for w in words:
                w = w.lower()
                for tok in tokens_to_replace:
                    w = w.replace(tok, '')
                w = w.replace('-', ' ')
                w = w.split(' ')
                for word in w:
                    if word in vectors:
                        glove_dict[word] = vectors[word]
                    else:
                        not_in_dict[word] = 1

    # check the words that are not in the Glove dictionary
    for k in not_in_dict.keys():
        print(k)

    print("Total number of words: {}".format(total_words))
    print("Number of glove: {}".format(len(glove_dict)))
    print("Number of words with no glove: {}".format(len(not_in_dict)))

    print("Dumping file...")
    pickle_dump(glove_dict, args.glove_out)
    
    print("Done!")



