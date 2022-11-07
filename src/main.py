import sys
import os
import json
import pandas as pd

import acronymDisambiguator
from utils import cleanText, cleanSentence
from semanticExpansion import expandSemantically    

OUTPUT_FOLDER = "../output/"

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

    # export the expanded text to a file called in the same way as the input
    outputLocation = OUTPUT_FOLDER + fileName
    open(outputLocation, "w").write(expandedText)


if __name__ == "__main__":
    ###ARGUMENTS###
    # 1: path to folder where input files are (.txts and .csvs)[common: ""../input"]
    # 2: name of dataset used ["scienceMed" or "science"]
    
    # Determine which model & dictionary to use
    
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

    # Make sure that the output folder exists. If not, create it.
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # for each file in the folder, run expandInputTextFile
    for file in os.listdir(pathToInputFolder):
        if file.endswith(".txt"):
            print("Expanding acronyms in " + file)
            expandInputTextFile(pathToInputFolder + "/" + file)
            print("Acronyms expanded in " + file)
            
        if file.endswith(".csv"):
            # read csv file
            inputScvFile = pd.read_csv(pathToInputFolder + "/" + file)
            print("Expanding acronyms in csv file" + file)
            total_acronyms = 0
            expanded_acronyms = 0
            # iterate over the rows of the csv file
            for index, row in inputScvFile.iterrows():
                total_acronyms += 1
                print("Expanding acronyms in sentence with id: " + str(row['id']))
                # get the text from the row
                text = row['text']
                # expand the acronyms in the text
                expandedText = expandAcronymInSentence(text)
                if not "No additional information found for" in expandedText:
                    expanded_acronyms += 1
                # add the expanded text in the same row in a new column
                inputScvFile.at[index, 'fullyExpandedText'] = expandedText
                print("Expanded text for sentence with id: " + str(row['id']) + " has been stored.")
                # export the expanded text to a file called in the same way as the input
                outputLocation = OUTPUT_FOLDER + file
                # store and export csv file
                inputScvFile.to_csv(outputLocation, index=False)
            print("Acronym expansion done. " + str(expanded_acronyms) + " out of " + str(total_acronyms) + " were expanded, this is " + str(expanded_acronyms/total_acronyms*100) + "%.")
                
                

