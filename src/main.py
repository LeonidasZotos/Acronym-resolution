import sys
import os
import json
import acronymDisambiguator

from utils import cleanText, cleanSentence
# from acronymDisambiguator import disambiguateAcronym
from semanticExpansion import expandSemantically    

def expandAcronymInSentence(sentence):
    originalSentence = sentence # save the original sentence so we can replace the acronym with the expanded version
    sentence = cleanSentence(sentence)
    
    for word in sentence: 
        # check if the word is an acronym and expand it if it is and it has no brackets
        if word.isupper() and len(word) > 1 and word.find("(") == -1: 
            # First, find which expansion is appropriate
            expandedAcronym = acronymDisambiguator.disambiguateAcronym(word, originalSentence)
            # Then, find the semantic expansion of the acronym
            semanticExpansion = expandSemantically(expandedAcronym)
            # Finally, replace the acronym with the acronym expansion and the semantic expansion
            fullExpansion = expandedAcronym + "(" + semanticExpansion + ")"
            # replace the acronym with the full expansion
            originalSentence = originalSentence.replace(word, fullExpansion)
    return originalSentence

def expandInputTextFile(pathToTextFile):
    # Get text from file
    text = open(pathToTextFile, "r").read()
    fileName = pathToTextFile.split("/")[-1]
    
    print("Cleaning sentences...")
    sentences = cleanText(text)
    print("Text is now split into sentences.")
    
    print("Expanding acronyms in sentences...")
    expandedText = [] # list of sentences with acronyms expanded
    for sentence in sentences:
        # append expanded sentence to expandedText
        expandedText.append(expandAcronymInSentence(sentence))

    # convert list of sentences to string
    expandedText = " ".join(expandedText)

    print("Acronyms expanded.")
    
    # make sure the output directory exists
    if not os.path.exists("output"):
        os.makedirs("output")

    # export the expanded text to a file called in the same way as the input
    outputLocation = "output/" + fileName
    open(outputLocation, "w").write(expandedText)


if __name__ == "__main__":
    ###ARGUMENTS###
    # 1: path to folder where input files are (.txts and .csvs)[common: ""../input"]
    # 2: name of dataset used ["scienceMed" or "science"]
    
    pathToInputFolder = sys.argv[1] 
    acronymDisambiguator.DATASET = sys.argv[2] # science or scienceMed
    if acronymDisambiguator.DATASET == 'science':
        acronymDisambiguator.MODEL_NAME = 'scienceModel.bin'
    elif acronymDisambiguator.DATASET == 'scienceMed':
        acronymDisambiguator.MODEL_NAME = 'scienceMedModel.bin'
    else:
        print('invalid model, exiting')
        exit()
    acronymDisambiguator.DICTIONARY = json.load(open('../' + acronymDisambiguator.DATASET + '/dict.json')) 

    
    # for each file in the folder, run expandInputTextFile
    for file in os.listdir(pathToInputFolder):
        if file.endswith(".txt"):
            print("Expanding acronyms in " + file)
            expandInputTextFile(pathToInputFolder + "/" + file)
            print("Acronyms expanded in " + file)
            
        if file.endswith(".csv"):
            print("Expanding acronyms in " + file)
            expandInputTextFile(pathToInputFolder + "/" + file)
            print("Acronyms expanded in " + file)