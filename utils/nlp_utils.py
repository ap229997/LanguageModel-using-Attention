# Additional utilities required

import numpy as np
from file_handlers import pickle_loader

class GloveEmbeddings(object):
    '''
    Class to get the glove embeddings of the words

    Arguments:
        glove_dim : dimension of the glove embedding

    Returns:
        None
    '''
    def __init__(self, file, glove_dim=300):
        self.glove = pickle_loader(file)
        self.glove_dim = glove_dim

    '''
    Get the glove embedding of the specified tokens

    Arguments:
        tokens : words whose glove embedding is required

    Returns:
        vectors : glove embedding of the specified tokens in vector form
    '''
    def get_embeddings(self, tokens):
        vectors = []
        for token in tokens:
            token = token.lower().replace("\'s", "")
            if token in self.glove:
                vectors.append(np.array(self.glove[token]))
            else:
                vectors.append(np.zeros((self.glove_dim,)))
        return vectors

# testing code
if __name__ == '__main__':
    glove = GloveEmbeddings('../data/glove_dict.pkl')
    emb = glove.get_embeddings(['and'])
    print (emb[0].shape)