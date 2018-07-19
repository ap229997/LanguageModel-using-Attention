# Variable Length LSTM network

import torch
import torch.nn as nn

'''
For variable length lstm, use padding with unknown '<unk>' token
'''
class VariableLengthLSTM(nn.Module):
	'''
	Class to define the variable length LSTM architecture

	Arguments:
		num_hidden : hidden dimemsion of the LSTM network
		depth : number of layers in the LSTM network
		word_emb_dim : word embedding dimension

	Returns:
		None
	'''
	def __init__(self, num_hidden=1024, depth=1, word_emb_dim=300):

		super(VariableLengthLSTM, self).__init__()

		self.num_hidden = num_hidden
		self.depth = depth
		self.word_emb_dim = 2*word_emb_dim # since we are concatenating glove embedding also

		self.lstm = nn.LSTM(input_size=self.word_emb_dim, hidden_size=self.num_hidden, num_layers=self.depth, batch_first=True, dropout=0)

	'''
	Computes forward pass through the LSTM network

	Arguments:
		word_emb : input word embedding

	Returns:
		out : lstm output
	'''
	def forward(self, word_emb):
		out = self.lstm(word_emb)

		return out
