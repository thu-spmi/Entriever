
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertForPreTraining, BertForNextSentencePrediction
from transformers import BertTokenizer
from reader import *
from metrics import *
from model import *
from evaluate import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, CrossEntropyLoss, CosineEmbeddingLoss, CosineSimilarity
from torch.utils.data import DataLoader

import os
import random
import logging
import json
from tqdm import tqdm
import numpy as np
import copy, re
from torch.utils.tensorboard import SummaryWriter
from config import global_config as cfg

def get_optimizers(num_samples, model, lr): # , cfg
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()],
            "weight_decay": 0.0,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    #print(num_samples, cfg.epoch_num, cfg.gradient_accumulation_steps, cfg.batch_size)
    num_training_steps = num_samples*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.batch_size)
    num_warmup_steps = int(num_training_steps*cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,\
        num_training_steps=num_training_steps)
    return optimizer, scheduler

def collate_fn_ebm(batch):
    collated = {}
    if batch!= []:
        for k,_ in batch[0].items():
            collated[k] = [b[k] for b in batch]
    return collated

def collate_fn(batch):
    pad_id = cfg.pad_id
    pad_result = {}
    label_shift = {1:0, -1:1}
    # label_shift2 = {1:1, -1:0}
    if cfg.only_one_model:
        wanted_key = ['input']
        for s  in batch:
            s['input'] = s['context'] + s['triple'][1:]
            sep_id = s['context'][-1]
    else:
        wanted_key = ['context', 'triple']
    if batch!= []:
        for key in wanted_key:# add padding for input, ouput and attentions
            #np.array(
            #attention=len(encoded)*[1]
            #if  not isinstance(self[0][key],int): 
            max_len = max(len(input[key]) for input in batch)
            max_len = min(max_len, cfg.max_sequence_len)
            pad_batch=np.ones((len(batch), max_len))*pad_id  #-100
            pad_attention_batch=np.ones((len(batch), max_len))*pad_id
            pad_type_batch=np.ones((len(batch), max_len))
            for idx, s in enumerate(batch):
                #trunc = s[-max_len:]
                if len(s[key])>cfg.max_sequence_len:
                    pad_batch[idx, :max_len] = np.array(s[key][-max_len:])
                    pad_attention_batch[idx, :max_len] = np.ones(max_len)
                else:
                    pad_batch[idx, :len(s[key])] = np.array(s[key])
                    pad_attention_batch[idx, :len(s[key])] = np.ones(len(s[key]))
                if cfg.only_one_model:
                    pad_type_batch[idx, :s[key].index(sep_id)] = np.ones(s[key].index(sep_id) if s[key].index(sep_id)<max_len else max_len)*0 # need more care afterwards
            pad_result[(key)] = torch.from_numpy(pad_batch).long()
            pad_result[(key+'_attention')] = torch.from_numpy(pad_attention_batch).long()
        if cfg.only_one_model:
            pad_result[(key+'_type')] = torch.from_numpy(pad_type_batch).long()
        
        if 'label' in batch[0]:
            pad_batch=np.ones(len(batch))
            for idx, s in enumerate(batch):
                #if cfg.only_one_model:
                pad_batch[idx] = label_shift[s['label']] # if cfg.only_one_model else s['label']
                #else:
                #   pad_batch[idx] = label_shift2[s['label']]
            pad_result['label'] = torch.from_numpy(pad_batch).long()
    return pad_result

# training code for baseline retrievers
def train(cfg, dataset='seretod'):
    cfg.exp_path = 'experiments_retrieve'
    cfg.batch_size = 16 # 32
    cfg.lr = 1e-5

    json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_path if dataset=='seretod' else cfg.bert_path_en)   
    # Add special tokens
    init_vocab_size=len(tokenizer)
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    logging.info('Added special tokens:{}'.format(special_tokens))
    tokenizer.add_special_tokens(special_tokens_dict)
    logging.info('Special token added, vocab size:{}-->{}'.format(init_vocab_size, len(tokenizer)))

    encoded_data = read_data(tokenizer, retrieve=True, dataset=dataset)
    if cfg.only_one_model:
        if dataset=='seretod':
            model = BertForNextSentencePrediction.from_pretrained('bert-base-chinese')#EncoderModel(cfg,tokenizer)
        else:
            model = BertForNextSentencePrediction.from_pretrained('bert-base')
        model.resize_token_embeddings(len(tokenizer))
        model.to(cfg.device[0])
    else:
        model = EncoderModel(cfg,tokenizer)

    train_dataloader=DataLoader(encoded_data['train'], batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn) 
    dev_dataloader=DataLoader(encoded_data['dev'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn)
    test_dataloader=DataLoader(encoded_data['test'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn)
   
    optimizer, scheduler = get_optimizers(num_samples=len(encoded_data['train']) ,model=model, lr=cfg.lr)
    global_step = 0
    metrics_to_log = {}
    best_score = -1
    for epoch in range(cfg.epoch_num):
        model.train()
        epoch_loss = 0
        epoch_step = 0
        num_batches = 0

        for batch in tqdm(train_dataloader):
            num_batches += 1

            # Transfer to gpu
            if torch.cuda.is_available():
                for key, val in batch.items():
                    if type(batch[key]) is list:
                        continue
                    batch[key] = batch[key].to(cfg.device[0])
                if cfg.only_one_model:
                    loss = model(input_ids=batch["input"], attention_mask=batch["input_attention"], token_type_ids=batch["input_type"],labels=batch["label"]).loss
                else:
                    loss = model(input_sent=batch["context"], attention_sent=batch["context_attention"], input_triple=batch["triple"], attention_triple=batch["triple_attention"],label=batch["label"])[0]
                loss.backward()
                epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                epoch_step += 1
                if epoch_step % cfg.gradient_accumulation_steps == 0 or num_batches==len(train_dataloader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

        logging.info("Epoch loss: {}".format(epoch_loss / num_batches))
        print(epoch_loss / num_batches)
        # Evaluate and save checkpoint
        score, precision, recall, f1 = evaluate(model, test_dataloader, cfg) # dev_dataloader
        metrics_to_log["eval_score"] = score
        logging.info("score: {}, precision: {}, recall: {}, f1: {}".format(score, precision, recall, f1))
        if dataset=='seretod':
            s = score + recall + f1
        else:
            s = score + f1
        print("score: {}, precision: {}, recall: {}, f1: {}".format(score, precision, recall, f1))
        if s > best_score:
            logging.info("New best results found! Score: {}".format(s))
            #model.bert_model.save_pretrained(cfg.save_dir)
            if cfg.only_one_model:
                save_path = cfg.bert_save_path if dataset=='seretod' else (cfg.bert_save_path+dataset)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
            else:
                if not os.path.exists(cfg.context_save_path):
                    os.mkdir(cfg.context_save_path)
                if not os.path.exists(cfg.triple_save_path):
                    os.mkdir(cfg.triple_save_path)
                if not os.path.exists(cfg.retrieval_save_path):
                    os.mkdir(cfg.retrieval_save_path)
                model.sentence_model.save_pretrained(cfg.context_save_path)
                model.triple_model.save_pretrained(cfg.triple_save_path)
                tokenizer.save_pretrained(cfg.context_save_path)
                tokenizer.save_pretrained(cfg.triple_save_path)
                torch.save(model.state_dict(), os.path.join(cfg.retrieval_save_path, "model.pt"))
            best_score = s
    #model.load_state_dict(torch.load(os.path.join(cfg.save_dir, "model.pt")))
    #score = evaluate(model, test_dataloader, cfg)
    #print(score)
    return

# training code for Entriever
def train_ebm(cfg, dataset='seretod'):
    # init logging handler
    cfg.exp_path = 'experiments_retrieve_ebm'
    if dataset=='seretod':
        model_name = f"model_allproposal{cfg.use_all_proposal}_mis_cache{cfg.train_ebm_mis}_residual{cfg.residual}"+ ("_add_feature" if cfg.add_extra_feature else "")
    else:
        model_name = f"model_allproposal{cfg.use_all_proposal}_mis_cache{cfg.train_ebm_mis}_residual{cfg.residual}_{dataset}"+ ("_add_feature" if cfg.add_extra_feature else "")
    init_logging_handler(model_name)

    cfg.lr = 5e-6
    if not os.path.exists(cfg.exp_path):
        os.mkdir(cfg.exp_path)
    json.dump(cfg.__dict__, open(os.path.join(cfg.exp_path,'cfg_all.json'), 'w'), indent=2)
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_path if dataset=='seretod' else cfg.bert_path_en)   
    # Add special tokens
    init_vocab_size=len(tokenizer)
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    logging.info('Added special tokens:{}'.format(special_tokens))
    # do
    tokenizer.add_special_tokens(special_tokens_dict)
    logging.info('Special token added, vocab size:{}-->{}'.format(init_vocab_size, len(tokenizer)))
    is_en = not dataset=='seretod'
    model = EBM(cfg, tokenizer, en=is_en)
    model.to(cfg.device[0])
    if cfg.only_one_model:
        save_path = cfg.bert_save_path if dataset=='seretod' else (cfg.bert_save_path+dataset)
        proposal_model = BertForNextSentencePrediction.from_pretrained(save_path) #EncoderModel(cfg,tokenizer)
        proposal_model.to(cfg.device[-1])
    encoded_data = read_data(tokenizer, ebm=True, dataset=dataset)
    if cfg.debugging:
        train_dataloader=DataLoader(encoded_data['train'][:400], batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn_ebm) 
    else:
        train_dataloader=DataLoader(encoded_data['train'], batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn_ebm)
    dev_dataloader=DataLoader(encoded_data['dev'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn_ebm)
    test_dataloader=DataLoader(encoded_data['test'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn_ebm)
   
    optimizer, scheduler = get_optimizers(num_samples=len(encoded_data['train']) ,model=model, lr=cfg.lr)
    global_step = 0
    metrics_to_log = {}
    best_score = -1
    nan_num = 0
    if dataset!='seretod':
        #kb_thresh_low = 0.1
        #kb_thresh_high = 0.9
        kb_thresh_low = 0.0
        kb_thresh_high = 1.0
        #kb_thresh = 15 # do not accept low prob result if kb result length is already over 15
        #kb_thresh_prob = 0.6
    else:
        kb_thresh_low = 0.0
        kb_thresh_high = 1.0
        #kb_thresh = 30
        #kb_thresh_prob = 0.5
    for epoch in range(cfg.epoch_num):
        statistic = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
        statistic_count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
        model.train()
        epoch_loss = 0
        gt_energy = 0
        epoch_step = 0
        num_batches = 0

        for batch in tqdm(train_dataloader):
            num_batches += 1

            # loss from data
            triples_gt = []

            try:  # avoid OOM
                positive_count = []
                for cases in batch['cases']:
                    gt_seq = ''
                    tmp_count = 0
                    for c in cases:
                        if c['label']==1:
                            #if dataset=='seretod':
                            gt_seq = gt_seq + tokenizer.decode(c['triple']).replace('[CLS]','').replace('[SEP]','') + '；'
                            #else:
                            #    gt_seq = gt_seq + tokenizer.decode(c['triple']).replace('[CLS]','').replace('[SEP]','')
                            tmp_count += 1 
                    if '；' in gt_seq:
                        gt_seq = gt_seq[:-1] 
                    triples_gt.append(gt_seq)
                    positive_count.append(tmp_count)
                gt_input = get_retrieval_sequence(tokenizer, batch['context'], triples_gt)
                gt_input.to(cfg.device[0])
                positive_counts = torch.tensor(positive_count, dtype=torch.float).to(cfg.device[0])
                if cfg.add_extra_feature:
                    logits = model(input_ids=gt_input['input_ids'], attention_mask=gt_input["attention_mask"], feature=positive_counts)
                else:
                    logits = model(input_ids=gt_input['input_ids'], attention_mask=gt_input["attention_mask"])
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    loss = 0.0
                    print(logits)
                    nan_num = nan_num + 1
                else:
                    loss = -sum(logits)
                    gt_energy += loss.item()
                    #if loss.item()>-2000:
                    #    print('warning, energy to large')
                    # loss = sum(math.exp(-logits)), incorrect as the loss is the -logp
                    # loss from MIS
                    # get_probability
                    statistic[0] += loss.item()
                    statistic_count[0] += cfg.batch_size # batch_size
                for i in range(len(batch['cases'])):
                    p_batch = collate_fn(batch['cases'][i])
                    for key, val in p_batch.items():
                        if type(p_batch[key]) is list:
                            continue
                        p_batch[key] = p_batch[key].to(cfg.device[-1])
                    if p_batch!={}:
                        with torch.no_grad():
                            p_logits = proposal_model(input_ids=p_batch["input"], attention_mask=p_batch["input_attention"], token_type_ids=p_batch["input_type"]).logits
                            probs = F.softmax(p_logits, dim=1)
                            accept_prob = probs[:,0].cpu().tolist() # 0 means coherency in bert pretraining
                            gt_label = p_batch['label'].cpu().tolist() 
                    else:
                        accept_prob = []
                    triple_num = len(accept_prob)
                    # sampling
                    if cfg.train_ebm_mis and 'cached' in batch['cases'][i][0]:
                        proposals = [batch['cases'][i][0]['cached'][0]]
                        proposal_log_probs = [batch['cases'][i][0]['cached'][1]]
                        proposal_wrong_num = [batch['cases'][i][0]['cached'][2]]
                    else:
                        proposals = []
                        proposal_log_probs = []
                        proposal_wrong_num = []
                    for sample_num in range(cfg.train_sample_num):
                        p_prob = 0.0
                        proposal = []
                        proposal_id = []
                        random.random()
                        for num in range(triple_num):
                            #kb_thresh_low
                            #kb_thresh_high
                            #if len(proposal)>kb_thresh:
                            #    p = kb_thresh_prob
                            #else:
                            p = random.random()
                            if (((p< accept_prob[num]) and (accept_prob[num]>kb_thresh_low)) or (accept_prob[num]>kb_thresh_high)):
                                if dataset=='seretod':
                                    proposal.append(tokenizer.decode(batch['cases'][i][num]['triple'][1:-1]).replace(' ','')) # can be unified to the .replace('[CLS]','').replace('[SEP]','')
                                else:
                                    proposal.append(tokenizer.decode(batch['cases'][i][num]['triple'][1:-1])) # can be unified to the .replace('[CLS]','').replace('[SEP]','')
                                # can directly concatenate all the triples to improve efficiency
                                p_prob += math.log(accept_prob[num])
                                proposal_id.append(0)
                            else:
                                p_prob += math.log(1-accept_prob[num])
                                proposal_id.append(1)
                        if proposal_id!=gt_label or cfg.use_all_proposal: #use cfg.use_all_proposal to include gt_label to be trained
                            #if dataset=='seretod':
                            proposals.append('；'.join(proposal))
                            #else:
                            #    proposals.append(','.join(proposal))
                            proposal_log_probs.append(p_prob)
                            proposal_wrong_num.append(sum(gt_label[i]!=proposal_id[i] for i in range(len(gt_label))))
                    # get IS_loss, avoiding OOM
                    is_logits = []
                    sample_num = len(proposals)
                    positive_count = torch.tensor([float(len(item.split('；'))) for item in proposals], 
                    dtype=torch.float).to(cfg.device[0])
                    if sample_num>0:
                        for b_num in range(cfg.train_sample_times): # cfg.train_sample_times>1 need to be modified
                            input = get_retrieval_sequence(tokenizer, [batch['context'][i]]*sample_num, proposals[b_num*16 : (b_num+1)*16])
                            input.to(cfg.device[0])
                            if cfg.add_extra_feature:
                                is_logits.extend(model(input_ids=input['input_ids'], attention_mask=input["attention_mask"], feature=positive_count))
                            else:
                                is_logits.extend(model(input_ids=input['input_ids'], attention_mask=input["attention_mask"]))
                        is_ratio = []
                        for j in range(sample_num):
                            if (-proposal_log_probs[j] + is_logits[j].item())>200:
                                is_ratio.append(math.exp(200))
                                print('large is ratio found')
                                nan_num = nan_num + 1
                            else:
                                if cfg.residual:
                                    is_ratio.append(math.exp (is_logits[j].item()))
                                else:
                                    is_ratio.append(math.exp(-proposal_log_probs[j] + is_logits[j].item()))
                        if cfg.train_ebm_mis:
                            mis_results = {}
                            max = is_ratio[0]
                            current = 0
                            lengths = 0
                            #mis_results[0] = 0
                            for j in range(sample_num):
                                tmp_prob = random.random()
                                if tmp_prob<(is_ratio[j]/max): # is_ratio[j]> max,
                                    # actually max is not necessarily the current max
                                    mis_results[current] = lengths
                                    max = is_ratio[j]
                                    current = j
                                    lengths = 1
                                else:
                                    lengths += 1
                            #if current==0:
                            #    mis_results[0] = lengths
                            mis_results[current] = lengths
                            # sample should be added
                            normalize = sum(mis_results[tmp] for tmp in mis_results) 
                            # mis performs averaging instead of weighted averaging
                            batch['cases'][i][0]['cached']=(proposals[j], proposal_log_probs[j], proposal_wrong_num[j])
                            # save cached results
                        else:
                            normalize = sum(is_ratio)
                        if normalize>0.0:
                            # mis sampling
                            if cfg.train_ebm_mis:    
                                for index, length_i in mis_results.items():
                                    if proposal_wrong_num[index] in statistic:
                                        statistic[proposal_wrong_num[index]] += is_logits[index].item()
                                        statistic_count[proposal_wrong_num[index]] += 1
                                    else:
                                        statistic[5] += is_logits[index].item()
                                        statistic_count[5] += 1
                                    loss = loss + (length_i*is_logits[index])/normalize
                            # is sampling
                            else:
                                for j in range(sample_num):
                                    if proposal_wrong_num[j] in statistic:
                                        statistic[proposal_wrong_num[j]] += is_logits[j].item()
                                        statistic_count[proposal_wrong_num[j]] += 1
                                    else:
                                        statistic[5] += is_logits[j].item()
                                        statistic_count[5] += 1
                                    """
                                    if cfg.reject_control:
                                        if is_ratio[j]/normalize>0.003: # add reject control
                                            loss = loss + (is_ratio[j]* is_logits[j])/normalize
                                        else:
                                            if random.random()<is_ratio[j]*200/normalize:
                                                loss = loss + 0.005*is_logits[j]
                                    else:
                                    """
                                    loss = loss + is_ratio[j]*is_logits[j]/normalize
                if loss!=0.0 and (not torch.isnan(loss).any()) and (not torch.isinf(loss).any()):
                    loss.backward()
                    epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                epoch_step += 1
                if epoch_step % cfg.gradient_accumulation_steps == 0 or num_batches==len(train_dataloader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    logging.info("WARNING: ran out of memory, batch size: {}".format(cfg.batch_size))
                    #if hasattr(torch.cuda, 'empty_cache'):
                    #    torch.cuda.empty_cache()
        logging.info("Epoch loss: {}".format(epoch_loss / num_batches))
        tmp = [statistic[i]/statistic_count[i] for i in statistic]
        print(tmp)
        print(f"epoch_loss: {epoch_loss / num_batches}")
        print(f"gt_energy: {gt_energy / num_batches}")
        # Evaluate and save checkpoint
        if not cfg.debugging:
            score, precision, recall, f1 = rescore(proposal_model, model, test_dataloader, cfg, tokenizer, kb_thresh_low=kb_thresh_low, 
        kb_thresh_high=kb_thresh_high) # dev_dataloader, kb_thresh=kb_thresh, kb_thresh_prob=kb_thresh_prob
            metrics_to_log["eval_score"] = score
            logging.info("score: {}, precision: {}, recall: {}, f1: {}".format(score, precision, recall, f1))
            s = score + f1 # recall +
            print("j-acc: {}, precision: {}, recall: {}, f1: {}".format(score, precision, recall, f1))
            #if (epoch-1)%5==0:
            #    score1, precision1, recall1, f11 = rescore(proposal_model, model, train_dataloader, cfg, tokenizer)
            #    print("j-acc: {}, precision: {}, recall: {}, f1: {}".format(score1, precision1, recall1, f11))
            ebm_save_path = cfg.ebm_save_path if dataset=='seretod' else (cfg.ebm_save_path+dataset)
            save_path = os.path.join(ebm_save_path, (model_name + ".pt"))
            if s > best_score:
                print(f"best checkpoints saved, score:{s}")
                #logging.info("New best results found! Score: {}".format(best_score))
                if not os.path.exists(ebm_save_path):
                    os.mkdir(ebm_save_path)
                tokenizer.save_pretrained(ebm_save_path)
                torch.save(model.state_dict(), save_path)
                best_score = s
        if nan_num>100:
            break
    #model.load_state_dict(torch.load(os.path.join(cfg.save_dir, "model.pt")))
    #score = evaluate(model, test_dataloader, cfg)
    #print(score)
    return

def init_logging_handler(model_name):
    stderr_handler = logging.StreamHandler()
    if not os.path.exists('./log'):
        os.mkdir('./log')
    file_handler = logging.FileHandler('./log/log_{}.txt'.format(model_name))
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

if __name__ == "__main__":
    # mode settings
    dataset = 'woz2.1' # seretod, camrest, incar, woz2.1
    cfg.only_one_model = True
    cfg.debugging = False
    #cfg.train_retrieve = False
    test_retrieve = False
    cfg.add_extra_feature = False
    if cfg.add_extra_feature:
        cfg.reg_weight = 15.0
    # ebm
    # cfg.train_ebm = False
    #cfg.train_retrieve = True
    cfg.train_ebm = True
    #cfg.train_retrieve = False
    if (not cfg.train_ebm) and (not cfg.train_retrieve):
        test_ebm = True
    else:
        test_ebm = False

    # define the exp name and init the logging handler
    cfg.exp_name = 'train_retrieve'
    cfg._init_logging_handler(dataset=dataset)

    cfg.use_all_proposal = True
    # mis settings
    cfg.train_ebm_mis = True # modified, MIS or not
    if cfg.train_ebm_mis:
        cfg.use_all_proposal = True # mis uses all proposals

    # other settings
    cfg.residual = True # residual or not
    cfg.reject_control = False # reject_control or not
    #cfg.device = [3, 6]
    #cfg.device = [6, 7]
    #cfg.device = [0]
    cfg.device = [0, 1]
    if cfg.train_ebm:
        # avoid out of memory issue
        if dataset=='seretod':
            cfg.batch_size = 6
        elif dataset=='camrest':
            cfg.batch_size = 4
        else:
            cfg.batch_size = 4
        cfg.eval_batch_size = 4
        train_ebm(cfg, dataset=dataset)
    """
    if cfg.train_ebm_mis:
        cfg.batch_size = 6
        cfg.eval_batch_size = 4
        train_ebm_with_mis(cfg)
    """
    if test_ebm:
        cfg.test_num = 16 # 8
        cfg.eval_batch_size = 8
        """
        if dataset!='seretod':
            kb_thresh = 15 # do not accept low prob result if kb result length is already over 15
            kb_thresh_prob = 0.6
        else:
            kb_thresh = 30
            kb_thresh_prob = 0.5
        """
        ebm_path = cfg.ebm_save_path + dataset # + '_thres' # cfg.ebm_save_path + dataset
        tokenizer = BertTokenizer.from_pretrained(cfg.ebm_save_path if dataset=='seretod' else ebm_path)
        is_en = not dataset=='seretod'
        model = EBM(cfg,tokenizer, en=is_en)
        # _thres
        save_path = os.path.join(cfg.ebm_save_path if dataset=='seretod' else ebm_path, 
        f"model_allproposal{cfg.use_all_proposal}_mis_cache{cfg.train_ebm_mis}_residual{cfg.residual}"+ ("_add_feature" if cfg.add_extra_feature else "") + ("" if dataset=='seretod' else ("_" + dataset)) + ".pt")
        model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
        model.to(cfg.device[0])
        if cfg.only_one_model:
            save_path_pro = cfg.bert_save_path if dataset=='seretod' else (cfg.bert_save_path+dataset)
            proposal_model = BertForNextSentencePrediction.from_pretrained(save_path_pro)#EncoderModel(cfg,tokenizer)
            proposal_model.to(cfg.device[-1])
        encoded_data = read_data(tokenizer, ebm=True, dataset=dataset)
        test_dataloader=DataLoader(encoded_data['test'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn_ebm)
        score, precision, recall, f1 = rescore(proposal_model, model, test_dataloader, cfg, tokenizer) # , kb_thresh=30, kb_thresh_prob=0.5
        print(save_path)
        print("j-acc: {}, precision: {}, recall: {}, f1: {}".format(score, precision, recall, f1))
    if cfg.train_retrieve:
        train(cfg, dataset=dataset)
    if test_retrieve:
        test(cfg, dataset)
