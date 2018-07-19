# Network Architecture

import torch
import torch.nn as nn
from torch.autograd import Variable

from var_len_lstm import VariableLengthLSTM
from attention import Attention

class Net(nn.Module):
	'''
	Class to define the network architecture

	Arguments:
		no_words : total count of words in the dataset
		lstm_size : numebr of hidden units in the lstm network
		emb_size : words embedding size
		depth : number of layers in the lstm network

	Returns:
		None
	'''
	def __init__(self, no_words=2627, lstm_size=1024, emb_size=300, depth=1):
		super(Net, self).__init__()

		self.word_cnt = no_words # to find total count of words - run create_dictionary.py in ../preprocess_data
		self.lstm_size = lstm_size
		self.emb_size = emb_size 
		self.depth = depth

		self.embedding = nn.Embedding(self.word_cnt, self.emb_size) # embedding for the words (this is different than the glove embedding)
		self.lstm = VariableLengthLSTM(num_hidden=self.lstm_size, depth=self.depth, word_emb_dim=self.emb_size)

		self.dropout = nn.Dropout(p=0.5)
		self.final_mlp = nn.Linear(self.lstm_size, self.word_cnt)

		self.attention = Attention(hidden_emb=self.lstm_size) # attention module

		self.softmax = nn.Softmax(dim=1)
		self.loss = nn.CrossEntropyLoss()

	'''
	Computes a forward pass through the network architecture

	Arguments:
		tokens : words in the sentence in tokenized form
		glove_emb : glove embedding of the words
		answer : label for each of the tokens (last word in the sentence is chosen to be the label)

	Returns:
		loss : cross entropy loss for the current batch
		ind : token of the predicted answer
	'''
	def forward(self, tokens, glove_emb, answer):

		####### Question Embedding #######
		# get the lstm representation of the final state at time t
		que_emb = self.embedding(tokens)
		emb = torch.cat([que_emb, glove_emb], dim=2)
		lstm_emb, internal_state = self.lstm(emb)

		####### Compute Attention ########
		attn_lstm_emb = self.attention(lstm_emb)
		batch_size = attn_lstm_emb.shape[0]
		attn_lstm_emb = attn_lstm_emb.view(batch_size, -1).contiguous()
		attn_lstm_emb = self.dropout(attn_lstm_emb)
		out = self.final_mlp(attn_lstm_emb)

		####### Compute softmax and loss ########
		prob = self.softmax(out)
		val, ind = torch.max(prob, dim=1)
		print ('predicted: ', ind.data, 'ground truth: ', answer.data)
		
		# hard cross entropy loss
		# answer as none implies validation
		if answer is not None:
			answer = answer.squeeze()
			loss = self.loss(out, answer)
			return loss, ind
		else:
			return ind

if __name__ == '__main__':
	net = Net()