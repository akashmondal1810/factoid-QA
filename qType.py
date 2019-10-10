# Hardcoded word lists
import sys
import numpy
import nltk
import collections
import json

yesnowords = ["can", "could", "would", "is", "does", "has", "was", "were", "had", "have", "did", "are", "will"]
commonwords = ["the", "a", "an", "is", "are", "were", "."]
questionwords = ["who", "what", "where", "when", "why", "how", "whose", "which", "whom"]


# Take in a tokenized question and return the question type and body
def processquestion(qwords):
    
    # Find "question word" (what, who, where, etc.)
    questionword = ""
    qidx = -1
    
    qw = [wo.lower() for wo in qwords if wo.lower() in questionwords]
    if len(qw)==0:
        return ("YESNO", qwords[1:])

    for (idx, word) in enumerate(qwords):
        if word.lower() in questionwords:
            questionword = word.lower()
            qidx = idx
            break

    if qidx < 0:
        return ("MISC", qwords)

    target = qwords[:qidx]+qwords[qidx+1:]
    
    type = "MISC"

    # Determine question type
    if questionword in ["who", "whose", "whom"]:
        type = "PERSON"
    elif questionword == "where":
        type = "PLACE"
    elif questionword == "when":
        type = "TIME"
    elif questionword == "which":
        type = "ITEM"
    elif questionword == "how":
        if target[0] in ["few", "little", "much", "many"]:
            type = "QUANTITY"
            target = target[1:]
        elif target[0] in ["young", "old", "long"]:
            type = "TIME"
            target = target[1:]

    # Trim possible extra helper verb
    if target[0] in yesnowords:
        target = target[1:]
    
    # Return question data
    return (type, target)
