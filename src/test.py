import pandas as pd
import sys
import json
import os
import numpy as np
import torch
import transformers
import tokenizers

from model import BertAD
OUTPUT_FOLDER = "../testFolder/output"
DATASET = ' ' # 'science' or 'scienceMed'
MODEL_NAME = ' ' # 'scienceModel.bin' or 'scienceMedModel.bin'
DICTIONARY =  { ' ' : [ ' ' ] } # Empty dictionary for now

def sample_text(text, acronym, max_len):
    text = text.split()
    idx = text.index(acronym)
    left_idx = max(0, idx - max_len//2)
    right_idx = min(len(text), idx + max_len//2)
    sampled_text = text[left_idx:right_idx]
    return ' '.join(sampled_text)

def process_data(text, acronym, expansion, tokenizer, max_len):

    text = str(text)
    expansion = str(expansion)
    acronym = str(acronym)

    n_tokens = len(text.split())
    if n_tokens>120:
        text = sample_text(text, acronym, 120)

    answers = acronym + ' ' + ' '.join(DICTIONARY[acronym])
    start = answers.find(expansion)
    end = start + len(expansion)

    char_mask = [0]*len(answers)
    for i in range(start, end):
        char_mask[i] = 1

    tok_answer = tokenizer.encode(answers)
    answer_ids = tok_answer.ids
    answer_offsets = tok_answer.offsets

    answer_ids = answer_ids[1:-1]
    answer_offsets = answer_offsets[1:-1]

    target_idx = []
    for i, (off1, off2) in enumerate(answer_offsets):
        if sum(char_mask[off1:off2])>0:
            target_idx.append(i)

    start = target_idx[0]
    end = target_idx[-1]


    text_ids = tokenizer.encode(text).ids[1:-1]

    token_ids = [101] + answer_ids + [102] + text_ids + [102]
    offsets =   [(0,0)] + answer_offsets + [(0,0)]*(len(text_ids) + 2)
    mask = [1] * len(token_ids)
    token_type = [0]*(len(answer_ids) + 1) + [1]*(2+len(text_ids))

    text = answers + text
    start = start + 1
    end = end + 1

    padding = max_len - len(token_ids)


    if padding>=0:
        token_ids = token_ids + ([0] * padding)
        token_type = token_type + [1] * padding
        mask = mask + ([0] * padding)
        offsets = offsets + ([(0, 0)] * padding)
    else:
        token_ids = token_ids[0:max_len]
        token_type = token_type[0:max_len]
        mask = mask[0:max_len]
        offsets = offsets[0:max_len]


    assert len(token_ids)==max_len
    assert len(mask)==max_len
    assert len(offsets)==max_len
    assert len(token_type)==max_len

    return {
            'ids': token_ids,
            'mask': mask,
            'token_type': token_type,
            'offset': offsets,
            'start': start,
            'end': end,  
            'text': text,
            'expansion': expansion,
            'acronym': acronym,
        }

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def evaluate_jaccard(text, selected_text, acronym, offsets, idx_start, idx_end):
    filtered_output = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += text[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            filtered_output += " "

    candidates = DICTIONARY[acronym]
    candidate_jaccards = [jaccard(w.strip(), filtered_output.strip()) for w in candidates]
    idx = np.argmax(candidate_jaccards)

    return candidate_jaccards[idx], candidates[idx]

def disambiguate(text, acronym):
    MODEL = BertAD()
    vec = MODEL.state_dict()['bert.embeddings.position_ids']
    chkp = torch.load(os.path.join('model', MODEL_NAME), map_location='cpu')
    chkp['bert.embeddings.position_ids'] = vec
    MODEL.load_state_dict(chkp)
    del chkp, vec
    TOKENIZER = tokenizers.BertWordPieceTokenizer(f"../" + DATASET + "/bert-base-uncased-vocab.txt", lowercase=True)
    MAX_LEN = 256


    inputs = process_data(text, acronym, acronym, TOKENIZER, MAX_LEN)
    ids = torch.tensor(inputs['ids'])
    mask = torch.tensor(inputs['mask'])
    token_type = torch.tensor(inputs['token_type'])
    offsets = inputs['offset']
    expansion = inputs['expansion']
    acronym = inputs['acronym']

    ids = torch.unsqueeze(ids, 0)
    mask = torch.unsqueeze(mask, 0)
    token_type = torch.unsqueeze(token_type, 0)

    start_logits, end_logits = MODEL(ids, mask, token_type)

    start_prob = torch.softmax(start_logits, axis=-1).detach().numpy()
    end_prob = torch.softmax(end_logits, axis=-1).detach().numpy()
    
    
    start_idx = np.argmax(start_prob[0,:])
    end_idx = np.argmax(end_prob[0,:])

    _, exp = evaluate_jaccard(text, expansion, acronym, offsets, start_idx, end_idx)
    return exp

def disambiguateAcronym(acronym, sentence):
    try:
        expansion = disambiguate(sentence, acronym)
        print("The expansion of ", acronym, "is: ", expansion)
    except:
        print("ERROR: Acronym", acronym, "not found in dictionary")
        return("NO EXPANSION FOUND")
    
    return expansion #return the expansion with the first letter capitalized


if __name__ == "__main__":
    ###ARGUMENTS###
    # 1: path to folder where test file is
    # 2: name of dataset used ["scienceMed" or "science"]
    
    # Determine which model & dictionary to use
    
    pathToTestFile = sys.argv[1] 
    DATASET = sys.argv[2] # science or scienceMed
    if DATASET == 'science':
        MODEL_NAME = 'scienceModel.bin'
    elif DATASET == 'scienceMed':
        MODEL_NAME = 'scienceMedModel.bin'
    else:
        print('invalid model, exiting')
        exit()
    DICTIONARY = json.load(open('../' + DATASET + '/dict.json')) 

    # Make sure that the output folder exists. If not, create it.
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    if pathToTestFile.endswith(".csv"):
        # read csv file
        testData = pd.read_csv(pathToTestFile)
        # First, generate all the expansions for the test set.
        print("Testing disambiguation of acronyms in file: " + pathToTestFile)
        # iterate over the rows of the csv file
        for index, row in testData.iterrows():
            print("Expanding acronyms in sentence with id: " + str(row['id']))
            # expand the acronyms in the text
            predictedExpansion = disambiguateAcronym(row['acronym_'], row['text'])
            # add the disambiguation in the same row in a new column
            testData.at[index, 'predictedExpansion'] = predictedExpansion
            print("Expanded text for sentence with id: " + str(row['id']) + " has been generated.")    
        # export the expanded text to a file called in the same way as the input
        outputLocation = OUTPUT_FOLDER +'/'+ DATASET + "Expanded.csv"
        # store and export csv file
        testData.to_csv(outputLocation, index=False)
        print("Output file has been generated at: " + outputLocation + ". Now evaluating generated expansions.")
        
        # Then, evaluate whether the generated expansions are correct.
        
        # for every row in testData, check if the predicted expansion is the same as the actual expansion
        correct = 0
        incorrect = 0
        for index, row in testData.iterrows():
            if row['predictedExpansion'] == row['expansion']:
                correct += 1
            else:
                incorrect += 1
        print("Correct expansions: " + str(correct))
        print("Incorrect expansions: " + str(incorrect))
        print("Accuracy: " + str(correct/(correct+incorrect)))
    else:
        print("ERROR: Input file is not a csv file. Please provide a csv file as input.")
        exit()
        