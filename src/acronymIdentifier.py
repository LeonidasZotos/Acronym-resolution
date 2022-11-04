import re

def expandAcronymInSentence(sentences):
    # list of acronyms and their contexts
    acronymsAndTheirSentences = []

    # for each sentence
    for sentence in sentences:
        # for each word
        for word in sentence:
            # if the word is an acronym, potentially with dashes or numbers
            if re.match(r'^[A-Z][A-Z0-9-]+$', word):
                # add the word and its context to the list
                acronymsAndTheirSentences.append((word, sentence))
    return acronymsAndTheirSentences