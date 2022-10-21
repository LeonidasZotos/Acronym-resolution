def expandAcronymInSentence(sentences):
    # list of acronyms and their contexts
    acronymsAndTheirSentences = []

    # for each sentence
    for sentence in sentences:
        # for each word
        for word in sentence:
            # if the word is an acronym
            if word.isupper():
                # add the acronym and its context to the list
                acronymsAndTheirSentences.append((word, sentence))

    return acronymsAndTheirSentences