import os
import sys
import json
import pandas as pd
import numpy as np
import pandas as pd
from typing import *
from tqdm.notebook import tqdm
from sklearn import model_selection
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import transformers
from transformers import AdamW
import tokenizers


def seed_all(seed = 42):
  """
  Fix seed for reproducibility
  """
  # python RNG
  import random
  random.seed(seed)

  # pytorch RNGs
  import torch
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

  # numpy RNG
  import numpy as np
  np.random.seed(seed)

class config:
  # dataset is the 1st argument. This is probably bad practice but here we are
  if len(sys.argv) > 1 and os.path.exists('../' + sys.argv[1]):
    dataset = sys.argv[1]
  else:
    print("Dataset not specified / found. Training on 'science' dataset")
    dataset = 'science'
    
  print("Loading dataset: ", dataset)
  TRAIN_FILE = '../' + dataset + '/train.csv'
  VAL_FILE = '../' + dataset + '/dev.csv'
  EXTERNAL_FILE = '../' + dataset + '/external_data.csv'
  bertLocation = '../' + dataset + '/bert-base-uncased'
  TOKENIZER = tokenizers.BertWordPieceTokenizer('../' + dataset + '/bert-base-uncased-vocab.txt', lowercase=True)
  DICTIONARY = json.load(open('../' + dataset + '/dict.json'))
  SEED = 42
  KFOLD = 3

  SAVE_DIR = '.'
  MAX_LEN = 192
  EPOCHS = 1
  TRAIN_BATCH_SIZE = 16
  VALID_BATCH_SIZE = 16
  
  A2ID = {}
  for k, v in DICTIONARY.items():
    for w in v:
      A2ID[w] = len(A2ID)
    
class AverageMeter:
    """
    Computes and stores the average and current value
    Source : https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch/
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping utility
    Source : https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch/
    """
    
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


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

  answers = acronym + ' ' + ' '.join(config.DICTIONARY[acronym])
  start = answers.find(expansion)
  end = start + len(expansion)

  char_mask = [0]*len(answers)
  for i in range(start, end):
    if(i < len(answers)):
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


class Dataset:
    def __init__(self, text, acronym, expansion):
        self.text = text
        self.acronym = acronym
        self.expansion = expansion
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        data = process_data(
            self.text[item],
            self.acronym[item],
            self.expansion[item], 
            self.tokenizer,
            self.max_len,
            
        )

        return {
            'ids': torch.tensor(data['ids'], dtype=torch.long),
            'mask': torch.tensor(data['mask'], dtype=torch.long),
            'token_type': torch.tensor(data['token_type'], dtype=torch.long),
            'offset': torch.tensor(data['offset'], dtype=torch.long),
            'start': torch.tensor(data['start'], dtype=torch.long),
            'end': torch.tensor(data['end'], dtype=torch.long),
            'text': data['text'],
            'expansion': data['expansion'],
            'acronym': data['acronym'],
        }


def get_loss(start, start_logits, end, end_logits):
  loss_fn = nn.CrossEntropyLoss()
  start_loss = loss_fn(start_logits, start)
  end_loss = loss_fn(end_logits, end)
  loss = start_loss + end_loss
  return loss


class BertAD(nn.Module):
  def __init__(self):
    super(BertAD, self).__init__()
    self.bert = transformers.BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    self.layer = nn.Linear(768, 2)
    

  def forward(self, ids, mask, token_type, start=None, end=None):
    output = self.bert(input_ids = ids,
                       attention_mask = mask,
                       token_type_ids = token_type)
    
    logits = self.layer(output[0]) 
    start_logits, end_logits = logits.split(1, dim=-1)
    
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)

    loss = get_loss(start, start_logits, end, end_logits)    

    return loss, start_logits, end_logits


def train_fn(data_loader, model, optimizer, device):
  model.train()
  losses = AverageMeter()
  tk0 = tqdm(data_loader, total=len(data_loader))
  
  for bi, d in enumerate(data_loader):
    
    ids = d['ids']
    mask = d['mask']
    token_type = d['token_type']
    start = d['start']
    end = d['end']
    
    ids = ids.to(device, dtype=torch.long)
    token_type = token_type.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    start = start.to(device, dtype=torch.long)
    end = end.to(device, dtype=torch.long)

    model.zero_grad()
    loss, start_logits, end_logits = model(ids, mask, token_type, start, end)

    loss.backward()
    optimizer.step()
    # xm.optimizer_step(optimizer, barrier=True)
    
    losses.update(loss.item(), ids.size(0))
    # tk0.set_postfix(loss=losses.avg)


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

  candidates = config.DICTIONARY[acronym]
  candidate_jaccards = [jaccard(w.strip(), filtered_output.strip()) for w in candidates]
  idx = np.argmax(candidate_jaccards)

  return candidate_jaccards[idx], candidates[idx]


def eval_fn(data_loader, model, device):
  model.eval()
  losses = AverageMeter()
  jac = AverageMeter()

  tk0 = tqdm(data_loader, total=len(data_loader))

  pred_expansion_ = []
  true_expansion_ = []

  for bi, d in enumerate(tk0):
    ids = d['ids']
    mask = d['mask']
    token_type = d['token_type']
    start = d['start']
    end = d['end']
    
    text = d['text']
    expansion = d['expansion']
    offset = d['offset']
    acronym = d['acronym']
    
    ids = ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    token_type = token_type.to(device, dtype=torch.long)
    start = start.to(device, dtype=torch.long)
    end = end.to(device, dtype=torch.long)
    
    
    
    with torch.no_grad():
      loss, start_logits, end_logits = model(ids, mask, token_type, start, end)


    start_prob = torch.softmax(start_logits, dim=1).detach().cpu().numpy()
    end_prob = torch.softmax(end_logits, dim=1).detach().cpu().numpy()
  
  
    jac_= []
    
    for px, s in enumerate(text):
      start_idx = np.argmax(start_prob[px,:])
      end_idx = np.argmax(end_prob[px,:])

      js, exp = evaluate_jaccard(s, expansion[px], acronym[px], offset[px], start_idx, end_idx)
      jac_.append(js)
      pred_expansion_.append(exp)
      true_expansion_.append(expansion[px])

    
    jac.update(np.mean(jac_), len(jac_))
    losses.update(loss.item(), ids.size(0))

    tk0.set_postfix(loss=losses.avg, jaccard=jac.avg)


  pred_expansion_ = [config.A2ID[w] for w in pred_expansion_]
  true_expansion_ = [config.A2ID[w] for w in true_expansion_]
  
  f1 = f1_score(true_expansion_, pred_expansion_, average='macro')

  print('Average Jaccard : ', jac.avg)
  print('Macro F1 : ', f1)

  return f1 
  
def run(df_train, df_val, fold):

  train_dataset = Dataset(
        text = df_train.text.values,
        acronym = df_train.acronym_.values,
        expansion = df_train.expansion.values
    )
  
  valid_dataset = Dataset(
        text = df_val.text.values,
        acronym = df_val.acronym_.values,
        expansion = df_val.expansion.values,
    )
    
  train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

  valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2
    )
  

  model = BertAD()
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  lr = 2e-5
  param_optimizer = list(model.named_parameters())
  no_decay = ['bias', 'gamma', 'beta']
  optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
      'weight_decay_rate': 0.01},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
      'weight_decay_rate': 0.0}
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

  es = EarlyStopping(patience=2, mode="max")

  print('Starting training....')
  for epoch in range(config.EPOCHS):
    train_fn(train_data_loader, model, optimizer, device)
    valid_loss = eval_fn(valid_data_loader, model, device)
    print(f'Fold {fold} | Epoch :{epoch + 1} | Validation Score :{valid_loss}')
    if fold is None:
      es(valid_loss, model, model_path=os.path.join(config.SAVE_DIR, "model.bin"))
    else:
      es(valid_loss, model, model_path=os.path.join(config.SAVE_DIR, f"model_{fold}.bin"))
    if es.early_stop:
      break

  return es.best_score


def run_k_fold(fold_id):
  '''
    Perform k-fold cross-validation
  '''
  seed_all()
  
  df_train = pd.read_csv(config.TRAIN_FILE)
  df_val = pd.read_csv(config.VAL_FILE)
  df_ext = pd.read_csv(config.EXTERNAL_FILE)
    
  # concatenating train and validation set
  train = pd.concat([df_train, df_val, df_ext]).reset_index()

  # dividing folds
  kf = model_selection.StratifiedKFold(n_splits=config.KFOLD, shuffle=True, random_state=config.SEED)
  for fold, (train_idx, val_idx) in enumerate(kf.split(X=train, y=train.acronym_.values)):
      train.loc[val_idx, 'kfold'] = fold


  print(f'################################################ Fold {fold_id} #################################################')
  df_train = train[train.kfold!=fold_id]
  df_val = train[train.kfold==fold_id]

  return run(df_train, df_val, fold_id)
    

if __name__ == "__main__":    
  # Do k-fold validation to find best model 
  f0 = run_k_fold(0)
  f1 = run_k_fold(1)
  f2 = run_k_fold(2)
  f3 = run_k_fold(3)
  f4 = run_k_fold(4)

  f = [f0, f1, f2, f3, f4]
  for i, fs in enumerate(f):
      print(f'Fold {i} : {fs}')
  print(f'Avg. {np.mean(f)}')