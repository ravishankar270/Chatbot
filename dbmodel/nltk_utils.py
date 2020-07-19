import nltk
import numpy as np
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(input_sentence, all_words):
    input_sentence= [stem(w) for w in input_sentence]
    print(input_sentence)
    bag= np.zeros(len(all_words), dtype=float)
    for i, w in enumerate(all_words):
        if w in input_sentence:
            bag[i]=1.0
    return bag

ip= ["How", "are", "you"]
words= ["are", "yo", "why", "hello"]
print(bag_of_words(ip, words))
