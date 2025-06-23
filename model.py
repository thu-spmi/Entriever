
from transformers import BertModel, BertForPreTraining, BertForNextSentencePrediction
import torch
import torch.nn as nn
from torch.nn import Dropout
import os, shutil
from torch.utils.tensorboard import SummaryWriter
from config import global_config as cfg
import json

# defination of the Entriever model
class EBM(torch.nn.Module):
    def __init__(self, cfg, tokenizer, en=False):
        super(EBM, self).__init__()
        self.cfg = cfg
        if en:
            self.bert_model=BertModel.from_pretrained('bert-base')
        else:
            self.bert_model=BertModel.from_pretrained('ret_exp/best_bert')
        # cfg.model_path
        # cannot connect to bert-base-chinese, use predownload version
        self.bert_model.resize_token_embeddings(len(tokenizer))
        self.dropout = Dropout(cfg.dropout)
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, 1) # we need to get an energy function, returning the logit
        if cfg.add_extra_feature:
            self.reg_weight = nn.Linear(1, 1) # cfg.reg_weight

    def forward(self,input_ids: torch.tensor,
                attention_mask: torch.tensor, feature=None):
        hidden_states = self.bert_model(input_ids=input_ids,attention_mask = attention_mask)[0] # B*L*h
        #pooled_output =  (hidden_states[:, :, :]*attention_mask.unsqueeze(-1)).mean(dim=1)
        # attention_mask.sum(dim=1)
        # a more reasonable energy definition which avoids different energy scale in different condition
        pooled_output =  (hidden_states[:, :, :]*attention_mask.unsqueeze(-1)).sum(dim=1)/attention_mask.sum(dim=1).unsqueeze(-1)
        logits = self.classifier(self.dropout(pooled_output))
        if feature is not None:
            logits = logits + self.reg_weight(feature.unsqueeze(-1))
            #for i in range(len(logits)):
            #    logits[i][0] = logits[i][0] + self.reg_weight(feature[i]) 
        #logits = self.classifier(pooled_output)
        return logits

# defination of dual-encoder baseline
class EncoderModel(torch.nn.Module):
    def __init__(self, cfg, tokenizer):
        super(EncoderModel, self).__init__()
        self.sentence_model=BertModel.from_pretrained('bert-base-chinese')
        self.triple_model=BertModel.from_pretrained('bert-base-chinese')
        self.inner_product = nn.Bilinear(self.sentence_model.config.hidden_size, self.triple_model.config.hidden_size, 1) # 768
        self.sentence_model.resize_token_embeddings(len(tokenizer))
        self.triple_model.resize_token_embeddings(len(tokenizer))
        if isinstance(cfg.device,list):
            self.device = cfg.device[0]
            self.device1 = cfg.device[-1]
        else:
            self.device = cfg.device
            self.device1 = cfg.device
        
        self.sentence_model.to(self.device)
        self.inner_product.to(self.device)
        self.triple_model.to(self.device1)

        if 'train' in cfg.mode:
            json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)        
        # log
        log_path='./log/log_retrieve/log_{}'.format(cfg.exp_name)
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
            os.mkdir(log_path)
        else:
            os.mkdir(log_path)
        self.tb_writer = SummaryWriter(log_dir=log_path)

    def forward(self,input_sent: torch.tensor,
                attention_sent: torch.tensor,
                input_triple,
                attention_triple,
                label):
        THRESHOLD = 0.5
        hidden_states = self.sentence_model(input_ids=input_sent,attention_mask = attention_sent)[0]
        h_sent =  hidden_states[:, :].mean(dim=1) # [cls] is also ok
        hidden = self.sentence_model(input_ids=input_triple,attention_mask = attention_triple)[0] # to triple's device
        h_triple =  hidden[:, :].mean(dim=1) # [cls] is also ok
        # cos_sim = CosineSimilarity(dim=1)
        # sim = cos_sim(h_sent, h_triple) # sim for score
        # cos_loss = CosineEmbeddingLoss() # margin = cfg.margin
        # loss = cos_loss(h_sent, h_triple, label) # can try Bilinear discriminator and celoss
        logits = self.inner_product(h_sent, h_triple) # need to be on the same device
        # loss_fct = nn.CrossEntropyLoss(reduction='sum') # 
        loss_fct = nn.BCELoss(reduction='sum')
        probs = torch.sigmoid(logits)
        # loss = nn.BCEWithLogitsLoss(logits, label.unsqueeze(-1).float())
        loss = loss_fct(probs, label.unsqueeze(-1).float())
        predictions = (probs<(1-THRESHOLD)).squeeze()
        labels = (label==0)
        accuracy = (predictions == labels).sum() / label.shape[0]
        
        return loss, accuracy.item(), predictions.cpu().tolist(), labels.cpu().tolist()

