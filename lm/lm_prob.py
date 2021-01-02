# -*- coding: utf-8 -*-

import math
import torch
import pickle
import numpy as np
from torch.autograd import Variable
import os
# from .model import RNNModel
# import model
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
class LMProb():

    def __init__(self, model_path, dict_path):
        with open(model_path, 'rb') as f:
            # self.model = RNNModel()
            self.model = torch.load(f)
            self.model.eval()
            # self.model = self.model.cpu()
        with open(dict_path, 'rb') as f:
            self.dictionary = pickle.load(f)
        print (len(self.dictionary))

    def get_prob(self, words, verbose=False):
        # print("entered")
        pad_words = ['<sos>'] + words + ['<eos>']
        indxs = [self.dictionary.getid(w) for w in pad_words]
        # print (indxs, self.dictionary.getid('_UNK'))
        with torch.no_grad():
            input = torch.LongTensor([int(indxs[0])]).unsqueeze(0).to('cuda:0')
            # print('entered')
            if verbose:
                print('words =', len(pad_words))
                print('indxs =', len(indxs))

            hidden = self.model.init_hidden(1).to('cuda:0')
            log_probs = []

            for i in range(1, len(pad_words)):
                # print(i)
                # print(input.device,hidden.device)
                output, hidden = self.model(input, hidden)
                # print (output.data.max(), output.data.exp())
                word_weights = output.squeeze().data.double().exp()
                # print (i, pad_words[i])
                # print(word_weights == None)
                prob = word_weights[indxs[i]] / word_weights.sum()

                prob = prob.to('cpu')

                log_probs.append(math.log(prob))
                # print('  {} => {:d},\tlogP(w|s)={:.4f}'.format(pad_words[i], indxs[i], log_probs[-1]))
                input.data.fill_(int(indxs[i]))

            if verbose:
                for i in range(len(log_probs)):
                    print('  {} => {:d},\tlogP(w|s)={:.4f}'.format(pad_words[i+1], indxs[i+1], log_probs[i]))
                print('\n  => sum_prob = {:.4f}'.format(sum(log_probs)))

            # return sum(log_probs) / math.sqrt(len(log_probs))
        # print((sum(log_probs)/len(log_probs)).device())
        return sum(log_probs) / len(log_probs)

