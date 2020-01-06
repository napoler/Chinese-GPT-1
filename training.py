# -*- coding: utf-8 -*-
"""Training.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bex9reeY-Xso2QojQgay17oeMmUkDtH8
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import time
import tqdm
import itertools
from allennlp.nn import util

from chinese_gpt import TransformerEncoder as Encoder
from chinese_gpt import TransformerDecoderLM as Decoder
from pytorch_pretrained_bert import BertModel, BertTokenizer, OpenAIAdam

encoder = Encoder()
decoder = Decoder()

encoder.load_state_dict(torch.load("model/encoder.pth"))
decoder.load_state_dict(torch.load("model/model_state_epoch_62.th"))


device = torch.device("cuda")
# device = torch.device("cpu")
#  a = torch.load("model/encoder.pth", map_location=torch.device('cpu'))
# encoder.load_state_dict(torch.load("model/encoder.pth", map_location=device))
# decoder.load_state_dict(torch.load("model/model_state_epoch_62.th", map_location=device))

train_data = torch.load("train_data.pth", map_location=device)

batch_size = 16
train_dataset = TensorDataset(*train_data)
train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)



encoder = encoder.to(device)
decoder = decoder.to(device)

num_epochs = 10
num_gradients_accumulation = 4
num_train_optimization_steps = num_train_optimization_steps = len(train_dataset) * num_epochs // batch_size // num_gradients_accumulation

param_optimizer = list(encoder.named_parameters()) + list(decoder.named_parameters()) 
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


optimizer = OpenAIAdam(optimizer_grouped_parameters,
                       lr=1e-5,
                       warmup=0.01,
                       max_grad_norm=1.0,
                       weight_decay=0.01,
                       t_total=num_train_optimization_steps)

update_count = 0
start = time.time()

for ep in range(num_epochs):
    
    pb = tqdm.tqdm_notebook(train_dataloader)
    
    for batch in pb:
        batch = [item.to(device) for item in batch]

        encoder_input, \
                third, \
                mask_encoder_input, \
                mask_third, \
                encoder_type_ids, \
                third_type_ids = batch
        
        _, past = encoder(encoder_input, mask_encoder_input, encoder_type_ids)
    
        mask = torch.cat([mask_encoder_input, mask_third], dim=1)
        logits, _ = decoder(third, mask, past=past, past_length=0)
        
        out = logits[:, :-1].contiguous()
        target = third[:, 1:].contiguous()
        target_mask = mask_third[:, 1:].contiguous()

        loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
        loss.backward()
        
        update_count += 1

        if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
            optimizer.step()
            optimizer.zero_grad()
            # speed measure
            end = time.time()
            speed = batch_size * num_gradients_accumulation / (end - start)
            start = end
            record_loss = loss.item()
            perplexity = np.exp(record_loss)

            pb.set_postfix(loss=record_loss, perplexity=perplexity, speed=speed)
    
    torch.save(encoder.state_dict(), str(ep)+"encoder.pth")
    torch.save(decoder.state_dict(), str(ep)+"decoder.pth")

torch.save(encoder.state_dict(), str(ep)+"encoder.pth")
torch.save(decoder.state_dict(), str(ep)+"decoder.pth")
# third_type_ids

