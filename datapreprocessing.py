import json
import pickle

import nltk
import numpy as np
from nltk.stem.lancaster import LancasterStemmer

import discord

# open json file
with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pkl", "w") as f:
        words, labels, traning, output = pickle.load(f)


except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)  # because wrds is already a list
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    stemmer = LancasterStemmer()
    # stem and do all elements lower in words list
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    # remove all duplicates
    words = sorted(list(set(words)))
    # sort labels
    labels = sorted(labels)

    # Neural networks only understand numbers
    # figure out the frequency of our words and put them in a list(One hot encoded)
    # This is really good input to our neural network so essentially determine what words are there what words aren't there.

    # This list gonna have a buch of  bags of words which are like just a list zeros and ones
    traning = []

    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        # Go through all of the different words thats are in our document or in wrds list
        # Put 0 or 1 into our bag of words depending on if it in the main world list.
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        traning.append(bag)
        output.append(output_row)

    # Turn these listes in numpy array
    traning = np.array(traning)  # len=41
    output = np.array(output)  # len

    pickle.dump(labels, open("labels.pkl", "wb"))
    with open("data.pkl", "wb") as f:
        pickle.dump((words, labels, traning, output), f)
