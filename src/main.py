import sys

from utils import cleanText, cleanSentence
from acronymDisambiguator import disambiguateAcronym
from semanticExpansion import expandSemantically    

def expandAcronymInSentence(sentence):
    originalSentence = sentence # save the original sentence so we can replace the acronym with the expanded version
    sentence = cleanSentence(sentence)
    
    for word in sentence: 
        # check if the word is an acronym and expand it if it is and it has no brackets
        if word.isupper() and len(word) > 1 and word.find("(") == -1: 
            # First, find which expansion is appropriate
            expandedAcronym = disambiguateAcronym(word, originalSentence)
            # Then, find the semantic expansion of the acronym
            semanticExpansion = expandSemantically(expandedAcronym)
            # Finally, replace the acronym with the acronym expansion and the semantic expansion
            fullExpansion = expandedAcronym + "(" + semanticExpansion + ")"
            # replace the acronym with the full expansion
            originalSentence = originalSentence.replace(word, fullExpansion)
    return originalSentence


if __name__ == "__main__":
    # Extract text from text file of the 1st argument
    text = open(sys.argv[1], "r").read()
    
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
    
    # export the expanded text to a file called output.txt
    open("output.txt", "w").write(expandedText)
