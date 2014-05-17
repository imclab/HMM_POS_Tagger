HMM_POS_Tagger
==============

Hidden Markov Model Part of Speech Tagger

This tagger implements smoothing on the start probabilities, emission probabilities, and transition probabilities (which are expected to be in python dictionary form, as learned in train.py). This is done in order to deal with words and tags/ tag transitions that were not seen in the training set. It then performs Viterbi decoding to find the most likely sequence of part of speech tags for the sentences in the test document.

train.py uses the NLTK POS tagger to tag all training documents in a directory specified at runtime, and extracts statistics based on these tags. It would be simple enough to change this script to use human-labeled POS tags (so that the accuracy is not limited by the accuracy of the NLTK POS tagger).
**Usage**

Learn the HMM based on some training documents in a given directory:
```
python train.py
```
You will be prompted to input the name of the directory which contains the training documents.

Use the model to tag a test document:
```
python test.py
```
You will be prompted to input the name of the test document, as well as the directory in which the HMM statistical models are stored.

Use the model to tag a test document and score accuracy:
```
python testAccuracy.py
```
