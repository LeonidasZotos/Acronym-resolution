import sys

from acronymIdentifier import acronymsAndContexts
from acronymDisambiguator import expandAcronyms
from semanticExpansion import expandSemantically

if __name__ == "__main__":
    # Extract text from text file of the 1st argument
    text = open(sys.argv[1], "r").read()
    
    acronymsAndTheirSentences = acronymsAndContexts(text)
    
    expandedText = expandAcronyms(text, acronymsAndTheirSentences)
    
    semanticExpText = expandSemantically(expandedText, acronymsAndTheirSentences)
    
    # export the expanded text to a file called output.txt
    open("output.txt", "w").write(semanticExpText)
