# dataloader to load the dataset in the format of input sequence (tokenized), their glove embeddings and labels (tokenized)

import logging
import os
import io
import sys
sys.path.insert(0,'../')
from utils.nlp_utils import GloveEmbeddings

from preprocess_data.tokenizer import Tokenizer

import torch
from torch.autograd import Variable

import numpy as np

logger = logging.getLogger()

# define the tokens to be replaced in the dataset (those beginning with 'u' represent unicode encoding format)
tokens_to_replace = [u"\u2019s", u"\u2019ve", u"\u2019t", u"\u2019", u"\u201c", u"\u201d", "(", ")", "[", "]", "{", "}",
					".", ",", "?", "*", "!", ";", ":", "_"]

'''
function to process the txt file into sequences of words

Arguments:
	path : path to the txt file

Return:
	processed_doc : processed txt file in the format of list of sequences
'''
def process_txt_file(path = '../data/train2.txt'):
	with io.open(path, 'r', encoding="utf8") as f:
		processed_doc = []
		for line in f:
			words = line.split()
			processed_words = []
			for word in words:
				word = word.lower()
				for token in tokens_to_replace:
					word = word.replace(token, "")
				word = word.replace('-', ' ')
				word = word.split(' ')
				for tok in word:
					processed_words.append(tok)
			if len(processed_words) > 1:
				processed_doc.append(processed_words)

	return processed_doc

'''
DataLoader : to load the 
				- Train set
				- Tokenizer
				- Glove Embeddings
'''
class DataLoader(object):

	def __init__(self, dict_path='data/dict.json', glove_path='data/glove_dict.pkl', data_path='data/train2.txt', batch_size=4, use_glove=True):
		super(DataLoader, self).__init__()

		self.batch_size = batch_size
		self.total_len = None # total size of the dataset being used - set real time
		self.use_glove = use_glove
		self.dict_path = dict_path
		self.glove_path = glove_path
		self.data_path = data_path

		# Load dictionary
		logger.info('Loading dictionary..')
		self.tokenizer = Tokenizer(self.dict_path)

		# Load data
		logger.info('Loading data..')
		self.trainset = process_txt_file(self.data_path)
		# can also load validation and test set if required, currently just loading the training set
		# self.valset = process_txt_file(self.data_path)
		# self.testset = process_txt_file(self.data_path)
		
		# Load glove
		self.glove = None
		if self.use_glove:
			logger.info('Loading glove..')
			self.glove = GloveEmbeddings(self.glove_path)

	'''
	function to get minibatch during training

	Arguments:
		ind : current iteration which is converted to required indices to be loaded
		data_type: specifies the train('train'), validation('val') and test('test') partition

	Returns:
		tokens : words in the train set in tokenized form
		glove_emb : glove embedding of the words
		answer : ground truth tokens (last word in the sentence to be predicted)
	'''
	def get_mini_batch(self, ind, data_type='train'):
		
		if data_type == 'train':
			dataset = self.trainset
		'''
		elif data_type == 'val':
			dataset = self.valset
		elif data_type == 'test':
			dataset = self.testset
		'''
		self.total_len = len(dataset) # total elements in dataset

		# specify the start and end indices of the minibatch
		# In case, the indices goes over total elements
		# wrap the indices around the dataset
		start_ind = (ind*self.batch_size)%self.total_len
		end_ind = ((ind+1)*self.batch_size)%self.total_len
		if start_ind < end_ind:
			data = dataset[start_ind:end_ind]
		else:
			data = dataset[start_ind:self.total_len]
			data.extend(dataset[0:end_ind])


		# get the sentences from the dataset, tokenize them and convert them into torch.autograd.Variable
		que = [x[:-1] for x in data]
		tokens = [self.tokenizer.encode_sentence(x)[0] for x in que]
		words = [self.tokenizer.encode_sentence(x)[1] for x in que]
		# max_len = max([len(x) for x in tokens]) # max length of the question
		max_len = 18 # checked from the dictionary to find the maximum length
		
		# pad the additional length with unknown token '<unk>'
		for x in tokens:
			for i in range(max_len-len(x)):
				x.append(self.tokenizer.word2i['<unk>'])
		for x in words:
			for i in range(max_len-len(x)):
				x.append('<unk>')
		tokens = Variable(torch.LongTensor(tokens))
		
		
		# get the ground truth answer, tokenize them and convert them into torch.autograd.Variable
		ans = [[x[-1]] for x in data]
		answer = [self.tokenizer.encode_sentence(x)[0] for x in ans]
		
		answer = Variable(torch.LongTensor(answer))
		
		# get the glove embeddings of the words token and convert them into torch.autograd.Variable
		glove_emb = [self.glove.get_embeddings(x) for x in words]
		glove_emb = Variable(torch.Tensor(glove_emb))

		return tokens, glove_emb, answer
	
# testing code
if __name__ == '__main__':
	data = DataLoader()

	tokens, glove_emb, answer = data.get_mini_batch(0)