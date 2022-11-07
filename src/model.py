import torch
import torch.nn as nn
import transformers

class BertAD(nn.Module):
  def __init__(self):
    super(BertAD, self).__init__()

    model_config = transformers.AutoConfig.from_pretrained('./model')
    model_config.update({"output_hidden_states":True})

    self.bert = transformers.BertModel(model_config)
    self.layer = nn.Linear(768, 2)
    

  def forward(self, ids, mask, token_type):
    output = self.bert(input_ids = ids,
                       attention_mask = mask,
                       token_type_ids = token_type)
    
    logits = self.layer(output[0]) 
    start_logits, end_logits = logits.split(1, dim=-1)
    
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)

    return start_logits, end_logits