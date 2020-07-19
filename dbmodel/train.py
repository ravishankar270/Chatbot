import json
import random
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
with open('intents.json','r',encoding='utf-8-sig') as f:
    intents=json.load(f)

#print(intents)
all_words=[]
tags=[]
xy=[]

for intent in intents['intents']:
    tag=  intent['Keyword']
    tags.append(tag)
    for qn in intent['question']:
        w= tokenize(qn)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words=['.', '?', ';', ',', '!']
all_words= [stem(w) for w in all_words if w not in ignore_words]
all_words= sorted(set(all_words)) #removes duplicates
tags=sorted(set(tags))
#print(tags)
#print(all_words)
#print(xy)

X_train=[]
Y_train=[]
train_data=[]
for (pattern, tag) in xy:
    print(pattern)
    print(all_words)
    
    bag= bag_of_words(pattern, all_words)
    print(bag)
    X_train.append(bag)
    label= tags.index(tag)
    Y_train.append(label)
X_train= np.array(X_train)

print(X_train)