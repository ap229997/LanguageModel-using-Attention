# contains the training script

import argparse
import logging
import os
import json

from model.net import Net
from dataloader import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import time

parser = argparse.ArgumentParser('Language model with attention')

parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--data_path", type=str, default='data/train2.txt', help="path of data")
parser.add_argument("--glove_path", type=str, default='data/glove_dict.pkl', help="path of glove")
parser.add_argument("--dict_path", type=str, default='data/dict.json', help="path of preprocessed dictionary")
parser.add_argument("--lstm_size", type=int, default=1024, help="number of hidden lstm units")
parser.add_argument("--use_glove", type=bool, default=True, help="whether to use glove")
parser.add_argument("--emb_size", type=int, default=300, help="embedding size")
parser.add_argument("--depth", type=int, default=1, help="number of lstm layers")
parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
parser.add_argument("--max_iters", type=int, default=50000, help="maximum iterations")
parser.add_argument("--batch_size", type=int, default=4, help="batch size for training")
parser.add_argument("--step_size", type=int, default=20000, help="number of iterations after which to decay learning rate")
parser.add_argument("--save_dir", type=str, default='saved_models', help="directory to save models")
parser.add_argument("--start_iter", type=int, default=0, help="load saved model from the given iteration")

args = parser.parse_args()

'''
Training script

Arguments:
	dataloader : to load the train set
	model : language model to be used
	optimizer : optimizer to be used

Returns:
	None
'''
def train(dataloader, model, optimizer):

	model.train() # define the state of the model
	iteration = args.start_iter

	while iteration < args.max_iters:
		st = time.time()
		tokens, glove_emb, answer = dataloader.get_mini_batch(iteration, data_type='train') # get minibatch input

		optimizer.zero_grad()
		loss, ans_token = model(tokens, glove_emb, answer) # get the loss and predicted answer
		loss.backward() # compute backpropagation
		optimizer.step() # update the weights

		iteration += 1

		print 'iteration: ', iteration, 'loss: ', loss.data[0], 'time taken: ', time.time()-st

		# save model state
		if iteration % 2000 == 0:
			torch.save(model.state_dict(), os.path.join(args.save_dir, 'iter_%d.pth'%(iteration)))

		# decrease learning rate by 10 after each step size
		if iteration % args.step_size == 0:
			lr = lr*0.1
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr


def main():
	torch.cuda.set_device(args.gpu)

	dataloader = DataLoader(dict_path=args.dict_path, glove_path=args.glove_path, data_path=args.data_path, batch_size=args.batch_size, use_glove=args.use_glove)

	model = Net(no_words=dataloader.tokenizer.no_words, lstm_size=args.lstm_size, emb_size=args.emb_size, depth=args.depth).cuda()

	if args.start_iter != 0:
		# load the model state from pre-specified iteration (saved model available)
		model.load_state_dict(torch.load(os.path.join(args.save_dir, 'iter_%d.pth'%(args.start_iter))), strict=False)
	
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	train(dataloader, model, optimizer)

if __name__ == '__main__':
	main()