#!/usr/bin/python

import sys
import os
#import operator # max(dictionary.iteritems(), key=operator.itemgetter(1))[0]
import errno
from math import log, exp
import fnmatch
import nltk
import pdb
import json

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def main():
    # Print awesome intro (Comment this out if you are not on a system that uses ansi escape characters for printing colors
    print bcolors.OKBLUE + " ___________  __    __    _______      ___      ___   __    ___  ___________  ______    _____  ___      "
    print bcolors.HEADER + '("     _   ")/" |  | "\\  /"     "|    |"  \\    /"  | |" \\  |"  |("     _   ")/    " \\  (\\"   \\|"  \\     '
    print bcolors.OKBLUE + " )__/  \\__/ (   (__)   )(  ______)     \\    \\  //  | ||  | ||  | )__/  \\\\__/// ____  \\ | \\   \\     |    "
    print bcolors.OKGREEN + "    \\_ /     \\/      \\/  \\/    |       /\\   \\/     | |   | |   |    \\_  /  /  /    )  )|  \\    \\   |    "
    print bcolors.WARNING + "    |   |    //  __  \\   // ___)_     |  \\         | |   |  \\  |___ |   | (  (____/ // |   \\    \\  |    "
    print bcolors.FAIL + '    \\   |   (   (  )   )(       "|    |   \\    /   | /\\  |\\( \\_|   \\    |  \\        /  |    \\    \\ |    '
    print bcolors.ENDC + '     \__|    \__|  |__/  \_______)    |___|\__/|___|(__\_|_)\_______)\__|   \\"_____/    \___|\____\)    '
    print bcolors.OKBLUE + '   _______    ______    ________      ___________   __       _______    _______    _______   _______    "                                                                                                        '
    print bcolors.ENDC + '  |   __ "\  /    " \  /"       )    ("     _   ") /""\     /" _   "|  /" _   "|  /"     "| /"      \   '
    print bcolors.FAIL + "  (  |__)  )// ____  \(    \___/      )__/  \\__/  /    \   (  ( \___) (  ( \___) (  ______) |        |  "
    print bcolors.WARNING + "  |   ____//  /    )  )\___  \           \\_  /   /' /\  \   \/ \       \/ \       \/    |   |_____/  )  "
    print bcolors.OKGREEN + "  (|  /   (  (____/ //  __/   \\          |   |  //  __'  \  //  \ ___  //  \ ___  // ___)_  //      /   "
    print bcolors.OKBLUE + ' /|__/ \   \        /  /" \    )         \   | /   /  \\   \ (    _( _|(    _(  _|(       "| |  __   \   '
    print bcolors.HEADER + '(_______)   \\"_____/  (_______/           \__|(___/    \___)\_______)  \_______)  \_______)|__|  \___)  '
    print bcolors.OKBLUE + "                                                                                                        "
    print bcolors.FAIL + " _______  ___  ___                                                                                      "
    print bcolors.FAIL + '|   _  "\\|"  \/"  |                                                                                     '
    print bcolors.FAIL + "(  |_)   )\   \  /                                                                                      "
    print bcolors.FAIL + "|      \/  \\  \/                                                                                       "
    print bcolors.FAIL + "(|  _  \\  /   /                                                                                        "
    print bcolors.FAIL + "|  |_)   )/   /                                                                                         "
    print bcolors.FAIL + "(_______/|___/                                                                                          "
    print bcolors.FAIL + "                                                                                                        "
    print bcolors.FAIL + " ________    _______   _______    _______  __   ___      ___________  ______    ____  ____  _______     "
    print bcolors.FAIL + '|"      "\  /"     "| /"      \\  /"     "||/"| /  ")    ("     _   ")/    " \  ("  _||_ " ||   _  "\    '
    print bcolors.FAIL + "(   ___   )(  ______)|         |(  ______)(  |/   /      )__/  \\\\__/// ____  \ |   (  )   |(  |_)   )   "
    print bcolors.FAIL + "|  \   ) || \/    |  |_____/   ) \/    |  |    __/          \\\\_ /  /  /    )  )(   |  |   )|      \/    "
    print bcolors.FAIL + "(| (___\\ || // ___)_  //      /  // ___)_ (// _  \          |   | (  (____/ //  |  \__/ / |(  _   \\    "
    print bcolors.FAIL + '|         )(       "||   __   \ (       "||  | \  \         \   |  \        /   ||  __  / ||  |_)   )   '
    print bcolors.FAIL + '(________/  \_______)|__|  \___) \_______)(__|  \__)         \__|   \\"_____/   (__________)(_______/    '
    print bcolors.ENDC + ""

    # Find all training documents
    inf = raw_input("Name the directory to search for training documents (enter a period for this directory): ")
    print inf
    matches = []
    for root, dirnames, filenames in os.walk(inf):
      for filename in fnmatch.filter(filenames, '*.txt'):
          matches.append(os.path.join(root, filename))
    # Make a sentence detector object
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    # Initialize dictionaries to hold transition and emission probabilities
    trans = {}
    emit = {}
    tagTot = {}
    lastForOne = {}
    startProb = {}
    numTags = 0.0
    tagTot["<s>"] = 0.0
    sentCount = 0.0
    print matches
    # Loop through all the training documents
    for f in matches:
        print "file is:"
        print f
        # Read in the text and tokenize
        text = open(f).read().strip()
        sentences = sent_detector.tokenize(text)
        words = [nltk.word_tokenize(s) for s in sentences]
        tagged = [nltk.pos_tag(w) for w in words]
        # Add sentence beginning tokens
        for i,t in enumerate(tagged):
            t.insert(0, ("<s>","<s>"))
        #print "tagged output of sentences in file:"
        #print tagged
        #pdb.set_trace()
        # Loop over each sentence in the document
        for s in tagged:
            #print "sentence is:"
            #print s
            i = 0
            le = len(s)
            # Loop over each word, tag pair in the sentence
            for word, tag in s[1:]:
                numTags += 1.0
                if i == le - 2:
                    #pdb.set_trace()
                    if tag in lastForOne:
                        lastForOne[tag] += 1.0
                    else:
                        lastForOne[tag] = 1.0
                #print "word and tag are:"
                #print word, tag
                if tag in tagTot:
                    #pdb.set_trace()
                    tagTot[tag] += 1.0
                else:
                    tagTot[tag] = 1.0
                # Dictionary of counts of tags given previous tag
                if s[i][1] in trans:
                    if tag in trans[s[i][1]]:
                        trans[s[i][1]][tag] += 1.0
                    else:
                        trans[s[i][1]][tag] = 1.0
                else:
                    trans[s[i][1]] = {}
                    trans[s[i][1]][tag] = 1.0
                # Dictionary of counts words given tags
                if tag in emit:
                    if word in emit[tag]:
                        emit[tag][word] += 1.0
                    else:
                        emit[tag][word] = 1.0
                else:
                    emit[tag] = {}
                    emit[tag][word] = 1.0
                i += 1
            sentCount += 1.0    # Keep track of the # of sentences processed
        print "Tag count total is:"
        print tagTot
        # Count the number of sentence start tags
        tagTot["<s>"] += len(tagged)
    # Divide all the counts by number of times the tag was seen to get probabilities
    for tag in emit:
        for word in emit[tag]:
            emit[tag][word] /= tagTot[tag]
            emit[tag][word] = log(emit[tag][word])
    for tagPrev in trans:
        for tag in trans[tagPrev]:
            if tagPrev in lastForOne:
                #pdb.set_trace()
                trans[tagPrev][tag] /= (tagTot[tagPrev] - lastForOne[tagPrev])
                trans[tagPrev][tag] = log(trans[tagPrev][tag])
            else:
                trans[tagPrev][tag] /= tagTot[tagPrev]
                trans[tagPrev][tag] = log(trans[tagPrev][tag])
    # Start probability of each tag is the probability of transition from sentence
    # beginning to that tag
    startProb = trans["<s>"]

    # Add in really small nonzero probabilities for all tags that never started a sentence
    #for tag in trans:
    #    if (tag not in startProb) and (tag != "<s>"):
    #        startProb[tag] = -500.0
    del trans["<s>"]

    for key in emit:
        summ = 0.0
        for ley in emit[key]:
            summ += exp(emit[key][ley])
        if summ > 1.0:
            print "WHOA BABY, THE SUM IS BIGGER THAN ONE FOR " + key + " " + str(summ)
            print summ
        else:
            print key + " " + str(summ)

    # Write statistics to a file, to separate training and testing
    of = raw_input("Name the file to store learned statistics: ")
    print of
    try:
        os.mkdir(of)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(of):
            pass
    f = open(of + "/emit.json","w")
    f.write(json.dumps(emit))
    f.close()
    f = open(of + "/trans.json","w")
    f.write(json.dumps(trans))
    f.close()
    f = open(of + "/tags","w")
    for key in emit:
        f.write(key + "\n")
    f.close()
    f = open(of + "/startProb.json","w")
    numTags += sentCount
    #for tag in tagTot:
    #    startProb[tag] = tagTot[tag]/numTags
    f.write(json.dumps(startProb))

if __name__ == "__main__":
    main()
