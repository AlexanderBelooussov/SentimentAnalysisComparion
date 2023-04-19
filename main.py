import csv

import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from models import ThreeClassTransformerModel, TransformerModel, VaderModel, TextBlobModel, FlairModel


def load_data(csv_path):
    """
    Load data from csv file.
    :param csv_path: path to csv file
    :return: list of lists (text, label)
    """
    with open(csv_path, 'r', encoding="ISO-8859-1") as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        data = list(reader)
    print(f"Loaded {len(data)} rows of data from {csv_path}")
    return data


def get_labels(data):
    """
    Get labels from data. Convert "positive" to 1 and "negative" to 0.
    :param data: list of lists (text, label)
    :return: np.array of labels
    """
    labels = [row[1] for row in data]
    labels = np.array(labels)
    labels = np.where(labels == "positive", 1, 0)
    return labels


def preprocess_text(data):
    """
    Preprocess text data. Remove quotes and <br /> tags.
    :param data: list of lists (text, label)
    :return: list of preprocessed text
    """
    texts = [row[0] for row in data]
    texts = [text.replace('"', '') for text in texts]
    texts = [text.replace('<br />', ' ') for text in texts]
    return texts


def main():
    csv_path = 'IMDB-movie-reviews.csv'
    data = load_data(csv_path)
    labels = get_labels(data)
    texts = preprocess_text(data)

    models = [
        TransformerModel(model_path="JiaqiLee/imdb-finetuned-bert-base-uncased"),
        ThreeClassTransformerModel(model_path="cardiffnlp/twitter-roberta-base-sentiment-latest"),
        TransformerModel(model_path="distilbert-base-uncased-finetuned-sst-2-english"),
        TransformerModel(model_path="aychang/roberta-base-imdb"),
        VaderModel(),
        TextBlobModel(),
        FlairModel(),
    ]

    scores = []
    times = []
    for model in models:
        predictions, t = model.timed_analyze(texts)
        score = accuracy_score(labels, predictions)
        scores.append(score)
        times.append(t)
        report = classification_report(labels, predictions)
        print(f"Model: {model.name}\n{report}")
        print(f"\n")

    # Print sorted scores
    max_name_len = max([len(model.name) for model in models])
    sorted_scores = sorted(zip(scores, times, models), reverse=True)
    print("\nSorted scores")
    print(f"{'Model':<{max_name_len}}\t{'Accuracy':<12}\t{'Time':<12}\t{'Time per text':<12}\t{'Params':<12}")
    print(f"{'-' * max_name_len:<{max_name_len}}\t{'-' * 12:<12}\t{'-' * 12:<12}\t{'-' * 12:<12}\t{'-' * 12:<12}")
    for score, t, model in sorted_scores:
        print(f"{model.name:<{max_name_len}}"
              f"\t{score:<12.4f}\t{t:<12.2f}"
              f"\t{t / len(texts):<12.2f}"
              f"\t{model.n_params if model.n_params is not None else '/':>12}")


if __name__ == '__main__':
    main()
