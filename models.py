import time

import numpy as np
import torch
from flair.data import Sentence
from flair.models import TextClassifier
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class BaseModel:
    """
    Base class for models.
    """

    def __init__(self):
        self.name = self.__class__.__name__
        self.n_params = None

    def analyze(self, texts):
        """
        Analyze texts and return the predictions.
        :param texts: list of texts
        :return: predictions
        """
        raise NotImplementedError

    def timed_analyze(self, texts):
        """
        Analyze texts and return the predictions and the time it took to run the model.
        :param texts: list of texts
        :return: predictions, time
        """
        start = time.time()
        predictions = self.analyze(texts)
        end = time.time()
        return predictions, end - start


class TransformerModel(BaseModel):
    """
    Class for transformer models.
    """

    def __init__(self, model_path):
        super().__init__()
        device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.pipeline = pipeline('sentiment-analysis', model=self.model, tokenizer=self.tokenizer, device=device)
        self.n_params = sum([p.numel() for p in self.model.parameters()])
        self.name = f"{model_path} ({self.n_params:,} params)"

        self.model.config.id2label = {0: 'negative', 1: 'positive'}
        self.model.config.label2id = {'negative': 0, 'positive': 1}

    def analyze(self, texts):
        predictions = self.pipeline(texts, truncation=True, max_length=512, top_k=1)
        predictions = [pred[0]['label'] for pred in predictions]
        predictions = np.where(np.array(predictions) == "positive", 1, 0)
        return predictions


class ThreeClassTransformerModel(TransformerModel):
    """
    Class for transformer models with 3 classes (positive, neutral, negative).
    """
    def __init__(self, model_path):
        super().__init__(model_path)
        self.model.config.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.model.config.label2id = {'negative': 0, 'neutral': 1, 'positive': 2}

    def analyze(self, texts):
        predictions = []
        for text in texts:
            pred = self.pipeline(text, truncation=True, max_length=512, top_k=None)
            pred = [pred for pred in pred if pred['label'] in ['positive', 'negative']]
            pred = sorted(pred, key=lambda x: x['score'], reverse=True)
            predictions.append(pred[0]['label'])
        predictions = np.where(np.array(predictions) == "positive", 1, 0)
        return predictions


class VaderModel(BaseModel):
    """
    Class for Vader model.
    """

    def __init__(self):
        super().__init__()
        self.model = SentimentIntensityAnalyzer()
        self.name = "Vader"

    def analyze(self, texts):
        sentiments = [self.model.polarity_scores(text)['compound'] for text in texts]
        sentiments = np.where(np.array(sentiments) > 0, 1, 0)
        return sentiments


class TextBlobModel(BaseModel):
    """
    Class for TextBlob model.
    """

    def __init__(self):
        super().__init__()
        self.model = TextBlob
        self.name = "TextBlob"

    def analyze(self, texts):
        sentiments = [self.model(text).sentiment.polarity for text in texts]
        sentiments = np.where(np.array(sentiments) > 0, 1, 0)
        return sentiments


class FlairModel(BaseModel):
    """
    Class for Flair model.
    """

    def __init__(self):
        super().__init__()
        self.model = TextClassifier.load('en-sentiment')
        self.n_params = sum([p.numel() for p in self.model.parameters()])
        self.name = f"Flair ({self.n_params:,} params)"

    def analyze(self, texts):
        predictions = []
        for text in texts:
            sentence = Sentence(text)
            self.model.predict(sentence)
            predictions.append(sentence.labels[0].value)
        predictions = np.where(np.array(predictions) == "POSITIVE", 1, 0)
        return predictions
