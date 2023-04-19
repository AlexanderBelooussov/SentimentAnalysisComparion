# Sentiment Analysis Comparison

A comparison of different sentiment analysis models on a dataset of IMDB movie reviews.

## Models
### Transformers
- [RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
  - Trained on Twitter data
- [RoBERTa](https://huggingface.co/"aychang/roberta-base-imdb)
  - Trained on IMDB data
- [DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
  - Trained on SST-2 data
- [BERT](https://huggingface.co/JiaqiLee/imdb-finetuned-bert-base-uncased)
  - Trained on IMDB data

### LSTM
- [Flair](https://github.com/flairNLP/flair)

### NLTK
- [VADER](https://github.com/cjhutto/vaderSentiment)
- [TextBlob](https://textblob.readthedocs.io/en/dev/)


## Usage
Create a virtual environment and install the requirements.
```python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Run the script.
```python
python main.py
```

## Results
```text
Model                                                                	Score       	Time        	Time per text	Params      
---------------------------------------------------------------------	------------	------------	------------	------------
JiaqiLee/imdb-finetuned-bert-base-uncased (109,483,778 params)       	0.9900      	23.38       	0.23        	   109483778
aychang/roberta-base-imdb (124,647,170 params)                       	0.9800      	25.72       	0.26        	   124647170
Flair (66,364,418 params)                                            	0.9200      	13.39       	0.13        	    66364418
distilbert-base-uncased-finetuned-sst-2-english (66,955,010 params)  	0.8900      	12.91       	0.13        	    66955010
cardiffnlp/twitter-roberta-base-sentiment-latest (124,647,939 params)	0.8500      	27.16       	0.27        	   124647939
Vader                                                                	0.6400      	0.40        	0.00        	           /
TextBlob                                                             	0.5900      	0.11        	0.00        	           /
```
### On a GPU
```text
Model                                                                	Time        	Time per text	Params      
---------------------------------------------------------------------	------------	------------	------------
JiaqiLee/imdb-finetuned-bert-base-uncased (109,483,778 params)       	1.74       	0.02        	   109483778
aychang/roberta-base-imdb (124,647,170 params)                       	1.77       	0.02        	   124647170
Flair (66,364,418 params)                                            	1.47       	0.01        	    66364418
distilbert-base-uncased-finetuned-sst-2-english (66,955,010 params)  	0.97       	0.01        	    66955010
cardiffnlp/twitter-roberta-base-sentiment-latest (124,647,939 params)	1.76       	0.02        	   124647939
```
(The first algorithm to run is always the slowest. For a fair comparison, experiment was run twice with different orders.)
## Note
Torch needs to be installed with GPU support to enable faster execution. For more information, see [here](https://pytorch.org/).