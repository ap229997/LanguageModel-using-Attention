# create a dictionary for the words in the dataset and tokenize them

import io
import json
import collections
import argparse

from nltk.tokenize import TweetTokenizer
from distutils.util import strtobool

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Creating dictionary..')

    parser.add_argument("-data_path", type=str, default='../data/train2.txt', help="Path to dataset")
    parser.add_argument("-dict_file", type=str, default="../data/dict.json", help="Name of the dictionary file")
    parser.add_argument("-min_occ", type=int, default=2, help='Minimum number of occurences to add word to dictionary')

    args = parser.parse_args()

    # additional tokens required
    word2i = {'<unk>': 0,
              '<start>': 1,
              '<stop>': 2,
              '<padding>': 3
              }

    # mapping words to their occurances in the train set
    word2occ = collections.defaultdict(int)

    # define the tokens to be replaced (those beginning with 'u' are in unicode format)
    tokens_to_replace = [u"\u2019s", u"\u2019ve", u"\u2019t", u"\u2019", u"\u201c", u"\u201d", "(", ")", "[", "]", "{", "}",
                    ".", ",", "?", "*", "!", ";", ":", "_"]

    # Input words
    tknzr = TweetTokenizer(preserve_case=False)
    max_len = 0
    path = args.data_path # path to the train set in txt format
    with io.open(path, 'r', encoding="utf8") as f:
        for line in f:
            words = line.split()
            max_len = max(max_len, len(words)) # find the maximum sequence length present in the train set
            for w in words:
                w = w.lower()
                for tok in tokens_to_replace:
                    w = w.replace(tok, '')
                w = w.replace('-', ' ')
                w = w.split(' ')
                for tok in w:
                    word2occ[tok] += 1

    included_cnt = 0
    excluded_cnt = 0
    for word, occ in word2occ.items():
        if occ >= args.min_occ and word.count('.') <= 1:
            included_cnt += occ
            word2i[word] = len(word2i)
        else:
            excluded_cnt += occ


    print("Number of words (occ >= {0:}): {1:} ~ {2:.2f}%".format(args.min_occ, len(word2i), 100.0*len(word2i)/len(word2occ)))
    print("Maximum sequence length present in the train set: {}".format(max_len))
    
    print("Dumping file...")
    with io.open(args.dict_file, 'wb') as f_out:
       data = json.dumps({'word2i': word2i})
       f_out.write(data.encode('utf8', 'replace'))
    
    print("Done!")

