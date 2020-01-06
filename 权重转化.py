import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


from chinese_gpt import TransformerEncoder as Encoder
from chinese_gpt import TransformerDecoderLM as Decoder
from pytorch_pretrained_bert import BertModel, BertTokenizer

from transformers import AlbertModel

encoder = Encoder()
decoder = Decoder()
# bert_model = BertModel.from_pretrained("bert-base-chinese")
bert_model = BertModel.from_pretrained("/mnt/data/dev/model/chinese_wwm_ext_pytorch")
# bert_model = AlbertModel.from_pretrained("/mnt/data/dev/github/albert_pytorch/albert_pytorch/albert_chinese_pytorch/prev_trained_model/albert_tiny")
# bert_model = AlbertModel.from_pretrained("/mnt/data/dev/github/albert_pytorch/albert_pytorch/albert_chinese_pytorch/prev_trained_model/albert_base_zh")
encoder_state_dict = encoder.state_dict()
# print(encoder_state_dict)
bert_state_dict = bert_model.state_dict()

for item in encoder_state_dict.keys():
    # print('item',item)
    if item in bert_state_dict:
        encoder_state_dict[item] = bert_state_dict[item]
        # print( bert_state_dict[item])
    else:
        print(item)
encoder.load_state_dict(encoder_state_dict)
torch.save(encoder.state_dict(), "model/v1/encoder.pth")

# decoder_state_dict = decoder.state_dict()
# temp_state_dict = torch.load("model_state_epoch_62.th")

# for item in decoder_state_dict.keys():
#     if item in temp_state_dict:
#         decoder_state_dict[item] = temp_state_dict[item]
#     else:
#         print(item)
        
# decoder.load_state_dict(decoder_state_dict)


# torch.save(decoder.state_dict(), "decoder.pth")