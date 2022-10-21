def cleanText(text):
    # returns a list of the words in the sentence
    # split text into sentences
    sentences = text.split(".")
    # remove empty strings
    sentences = [sentence for sentence in sentences if sentence != ""]
    # remove newlines
    sentences = [sentence.replace("\n", " ") for sentence in sentences]
    # remove leading and trailing spaces
    sentences = [sentence.strip() for sentence in sentences]    
    return sentences

def cleanSentence(sentence):
    # returns a list of the words in the sentence
    # remove leading and trailing spaces in each word
    # separate sentence into words
    sentence = sentence.split(" ")
    # remove empty strings
    sentence = [word for word in sentence if word != ""]

    return sentence