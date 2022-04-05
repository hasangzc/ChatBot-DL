import pickle
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from model import model
from datapreprocessing import words, data


def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    # ERROR_THRESHOLD = 0.99
    results = [[i, r] for i, r in enumerate(res)]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        labels = pickle.load(open("labels.pkl", "rb"))
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
    return return_list


def chat(json_file):
    print("HHG-BOT ile konuşmaya başla. quit yazarsan gider:)")
    while True:
        inp = input("Person: ")
        if inp.lower() == "quit":
            break

        p = bow(inp, words)
        results = model.predict(np.array([p]))[0]
        # results = model.predict([bag_of_words(inp, words)])
        results = [[i, r] for i, r in enumerate(results)]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            labels = pickle.load(open("labels.pkl", "rb"))
            return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
        list_of_intents = json_file["intents"]
        # print(list_of_intents)
        for i in list_of_intents:
            ints = return_list

            tag = ints[0]["intent"]
            result = "null"
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                print(result)
        return result


# chat(json_file=data)
