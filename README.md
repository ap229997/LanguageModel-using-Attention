## LanguageModel-using-Attention
Pytorch implementation of a basic language model using Attention in LSTM network

### Introduction
This repository contains code for a basic language model to predict the next word given the context. The network architecture used is LSTM network with Attention. The sentence length can be variable and this is taken care by padding the additional steps in the sequence. The model is trained using text from the book *The Mercer Boys at Woodcrest by Capwell Wyckoff* available at [http://www.gutenberg.org](http://www.gutenberg.org). Any other ebook or txt from other sources can also be used for training the network.

### Setup
This repository is compatible with python 2. </br>
- Follow instructions outlined on [PyTorch Homepage](https://pytorch.org/) for installing PyTorch (Python2). 
- The python packages required are ``` nltk ``` which can be installed using pip. </br>

### Data
Download any ebook available at [http://www.gutenberg.org](http://www.gutenberg.org) in ```.txt``` format. Create a new directory ```data``` and store the txt file in it. Any other text source can also be used.

### Process Data
The txt file is first preprocessed to remove some unwanted tokens, filter rarely used words and converted into dictionary format. In addition the glove embeddings are also to be loaded.

#### Create dictionary
To create the dictionary, use the script ```preprocess_data/create_dictionary.py``` </br>
```
python create_dictionary.py --data_path path_to_txt_file --dict_file dict_file_name.json --min_occ minimum_occurance_required
```

#### Create GLOVE dictionary
To create the GLOVE dictionary, download the original glove file and run the script ```preprocess_data/create_gloves.py```
```
wget http://nlp.stanford.edu/data/glove.42B.300d.zip -P data/
unzip data/glove.42B.300d.zip -d data/
python preprocess_data/create_gloves.py --data_path path_to_txt_file --glove_in data/glove.42B.300d.txt --glove_out data/glove_dict.pkl
```
If there is an issue in downloading using the script, then the glove file can be downloaded from [here](https://drive.google.com/open?id=1useknMuCENHTbMvSx0-Nmlau45XiIquF).

### Train the model
To train the model, run the following script </br>
```
python main.py --gpu gpu_id_to_use --use_cuda True --data_path path_to_txt_file --glove_path data/glove_dict.pkl --dict_path path_to_dict_file
```
The other parameters to be used are specified in ```main.py```. Refer to it for better understanding. </br>
The saved models are available [here](https://drive.google.com/drive/folders/1PRztMkrBe8bRyjS7HaW5DbTh6TrFW3-Z?usp=sharing).
