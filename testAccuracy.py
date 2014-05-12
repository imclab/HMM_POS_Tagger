#!/usr/bin/python

import sys
import os
#import operator # max(dictionary.iteritems(), key=operator.itemgetter(1))[0]
import errno
from math import log
import fnmatch
import nltk
import pdb
import json

 
def evaluate(vitPath, nltkPath, confuse):
    for n, tag in enumerate(vitPath):
        tmp = nltkPath[n][1]
        # Shouldn't ever be an issue, but just in case
        if tmp in confuse[tag]:
            confuse[tag][tmp] += 1
        else:
            confuse[tag][tmp] = 1
    return confuse

# Python Viterbi from Wikipedia, with added error checking and smoothing
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
    lenS = len(states)
    # Initialize base cases (t == 0)
    for y in states:
        # Deal with cases in which no data is stored for the tag of interest
        # Give a REALLY small nonzero likelihood (this makes things not true likelihoods anymore unless I subtract from the stored
        if y not in start_p:
            a = -500.0
        else:
            a = start_p[y]
        if y not in emit_p:
            b = -500.0
        elif obs[0] not in emit_p[y]:
            b = -500.0
        else:
            b = emit_p[y][obs[0]]
        V[0][y] = a + b
        path[y] = [y]
 
    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            prob = -10000.0
            state = states[0]

            for y0 in states:
                # Deal with cases in which no data is stored for the tag of interest

                # If y0 never transitioned to anything in the training set 
                # (could happen if it was always at the end of the sentence)
                if y0 not in trans_p:
                    a = -500.0
                # If y0 never transitioned to y in the training set
                elif y not in trans_p[y0]:
                    a = -500.0
                else:
                    a = trans_p[y0][y]
                # If that tag was never seen.. shouldn't ever happen..
                if y not in emit_p:
                    print "Ended up with a tag in 'tags' that was never seen in training.."
                    print y
                    b = -500.0
                # If the word was never observed with this tag
                elif obs[t] not in emit_p[y]:
                    # Give a slightly higher emission probility to the noun tags 
                    # for unknown words, as nouns are used very often
                    if y == "NN" or y == "NNS":
                        b = -499.0
                    else:
                        b = -500.0
                else:
                    b = emit_p[y][obs[t]]
                c = V[t-1][y0] + a + b
                if c > prob:
                    prob = c
                    state = y0     
            #(prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]
 
        # Don't need to remember the old paths
        path = newpath
    # if only one element is observed max is sought in the initialization values
    n = 0
    if len(obs) != 1:
        n = t
    #print_dptable(V)
    (prob, state) = max((V[n][y], y) for y in states)
    return (prob, path[state])
 
# This prints a table of the Viterbi steps
def print_dptable(V):
    s = "    " + " ".join(("%7d" % i) for i in range(len(V))) + "\n"
    for y in V[0]:
        s += "%.5s: " % y
        s += " ".join("%.7s" % ("%f" % v[y]) for v in V)
        s += "\n"
    print(s)

def main():
    # Make a sentence detector object
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    #states = []
    tagSet = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB', '.', ',', ':', '(', ')','``',"''",'-NONE-'] 
    g = 1
    # Get the statistics from wherever they are saved
    while(1):
        inf = raw_input("WHERE ARE THE STATS? WHERE ARE THEY?")
        print inf
        try:
            f = open(str(inf) + "/emit.json")
            f.close()
            g = 0
        except:
            print "I NEED TO KNOW THE TRUTH"
            g = 1
        if g == 0:  
            break
    f = open(str(inf) + "/emit.json").read()
    emit = json.loads(f)
    f = open(str(inf) + "/trans.json").read()
    trans = json.loads(f)
    tags = open(str(inf) + "/tags").read().splitlines()
    f = open(str(inf) + "/startProb.json").read()
    startP = json.loads(f)
    
    #pdb.set_trace()
    g = 1
    # Get the statistics from wherever they are saved
    while(1):
        inf = raw_input("TEST FILE?")
        print inf
        try:
            f = open(inf)
            f.close()
            g = 0
        except:
            print "NO"
            g = 1
        if g == 0:  
            break
    text = open(inf).read().strip()
    sentences = sent_detector.tokenize(text)
    words = [nltk.word_tokenize(s) for s in sentences]
    #tagged = [nltk.pos_tag(w) for w in words]
    # Add sentence beginning tokens  
    #for i,t in enumerate(words):
    #    t.insert(0, "<s>")  
    #pdb.set_trace()

    # Testing time!

    # Initialize the confusion matrix, which will be implemented using a dictionary for speed
    confuse = {}
    for predicted in tagSet:
        confuse[predicted] = {}
    for actual in tagSet:
        confuse[predicted][actual] = 0
            
    # Evaluate the path using MILTON POS TAGGER as well as NLTK POS TAGGER
    for s in words:
        prob, path = viterbi(s, tags, startP, trans, emit)
        #print s
        #print path
        nltkPath = nltk.pos_tag(s)
        #print nltkPath
        confuse = evaluate(path, nltkPath, confuse)

    #pdb.set_trace()

    ## Print out the results (confusion matrix) [it's really big so I just commented this out]
    #print "Mapping between tags and indices in confusion matrix:"
    #behold = []
    #for n, tag in enumerate(tagSet):
    #    behold.append((n,tag))
    #print behold   
    #tagLen = len(tagSet)
    #print range(tagLen)
    #for n, actual in enumerate(tagSet):
    #    printS = str(n) + " "
    #    for predicted in tagSet:
    #        # Again, should never be a problem, but just being safe
    #        if actual in confuse[predicted]:
    #            printS += str(confuse[predicted][actual]) + " "
    #        else:
    #            printS += "0 "
    #    print printS

    # Get the overall accuracy for each tag
    sTot = 0.0
    sCor = 0.0
    print "Accuracy for individual tags:"
    for actual in tagSet:
        total = 0.0
        if actual in confuse[actual]:
            correct = confuse[actual][actual]
        else:
            correct = 0.0
        for predicted in tagSet:
            if actual in confuse[predicted]:
                total += confuse[predicted][actual]
        # Don't divide by zero, kids.
        sCor += correct
        if total != 0.0:
            correct /= total
            printS = actual + " " + str("%10.1f" % (100*correct)) + "% of " + str(int(total))
        else:
            printS = actual + " N/A % (never tagged)"
        print printS
        sTot += total
    print "\n"
    print "System overall accuracy:"
    if sTot != 0.0:
        sCor /= sTot
        print str("%10.2f" % (100*sCor)) + "% of " + str(int(sTot)) + " tags correct"
    else:
        print "Nothing tagged?"    
        

if __name__ == "__main__":
    main()
