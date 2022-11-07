import json
import os
import numpy as np
import torch
import transformers
import tokenizers

from model import BertAD
DATASET = ' ' # can also be 'scienceMed'
MODEL_NAME = ' ' # can also be 'scienceMedModel.bin'
DICTIONARY =  { ' ' : [ ' ' ] }

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

    print("Dataset used is called: ", DATASET)
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
    
    return expansion.title() #return the expansion with the first letter capitalized