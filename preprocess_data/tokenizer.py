# Tokenize the words in the dataset

from nltk.tokenize import TweetTokenizer
import json
import re

class Tokenizer:
    '''
    Class to tokenize the words present in the dataset

    Arguments:
        dictionary_file : dictionary file containing the words in the dataset and their frequency of occurances

    Returns:
        None
    '''
    def __init__(self, dictionary_file):

        self.tokenizer = TweetTokenizer(preserve_case=False)
        with open(dictionary_file, 'r') as f:
            data = json.load(f)
            self.word2i = data['word2i']

        self.dictionary_file = dictionary_file

        self.i2word = {}
        for (k, v) in self.word2i.items():
            self.i2word[v] = k
        
        # Retrieve key values
        self.no_words = len(self.word2i)

        self.unknown_question_token = self.word2i["<unk>"]
        self.padding_token = self.word2i["<unk>"]

    '''
    function to tokenize the words (sentence refers to list of words)

    Arguments:
        words : words to be tokenized in list format

    Returns:
        tokens : tokens corresponding to the input words in list format
        words : input words in list format
    '''
    def encode_sentence(self, words):
        tokens = []
        for token in words:
            if token not in self.word2i:
                token = '<unk>'
            tokens.append(self.word2i[token])
            
        # return both the tokens and list of words
        return tokens, words

    '''
    function to convert the list of tokens into their corresponding words to form a sentence

    Arguments:
        tokens : tokens to be converted into words in list format

    Returns:
        words : converted tokens in list format
    '''
    def decode_sentence(self, tokens):
        words = ' '.join([self.i2word[tok] for tok in tokens]) 
        return words
    

# testing code
if __name__ == '__main__':
    tokenizer = Tokenizer('../data/dict.json')

    tokens, words = tokenizer.encode_sentence(['and'])
    print tokens, words
    sentence = tokenizer.decode_sentence([502])
    print sentence