

def acronymAndContexts(text):
    # returns the acronyms and their contexts in a list of tuples
    # where each tuple is (acronym, context)
    # context is a list of sentences where the acronym appears

    # split text into sentences
    sentences = text.split(".")
    # remove empty strings
    sentences = [sentence for sentence in sentences if sentence != ""]
    # remove newlines
    sentences = [sentence.replace("\n", " ") for sentence in sentences]
    # remove leading and trailing spaces
    sentences = [sentence.strip() for sentence in sentences]
    # remove leading and trailing spaces in each word
    sentences = [sentence.split(" ") for sentence in sentences]
    # remove empty strings
    sentences = [[word for word in sentence if word != ""] for sentence in sentences]
    
    # list of acronyms and their contexts
    acronymsAndTheirSentences = []

    # for each sentence
    for sentence in sentences:
        # for each word
        for word in sentence:
            # if the word is an acronym
            if word.isupper():
                # add the acronym and its context to the list
                acronymsAndTheirSentences.append((word, sentences))

    return acronymsAndTheirSentences