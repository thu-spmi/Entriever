from transformers import BertForNextSentencePrediction
from transformers import BertTokenizer
from reader import *
from metrics import *
from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import numpy as np
import copy, re
from torch.utils.tensorboard import SummaryWriter
from config import global_config as cfg

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

# getting retrieval results for entriever
def rescore(proposal_model, model, val_dataloader, hparams, tokenizer, kb_thresh_low=0.0, 
        kb_thresh_high=1.0): # tokenizer, kb_thresh=30, kb_thresh_prob=0.5
    origin_threshold = 0.5 # means the sampling probability increased, discard used in sampling
    accept_threshold = kb_thresh_high # used to simplify computation, accept those triples above the Threshold and ignore them
    reject_threshold = kb_thresh_low # used to simplify computation, discard those triples above the Threshold and ignore them
    topk_num = cfg.test_num
    model.eval()
    joint_acc = 0
    joint_acc_e = 0
    joint_acc_ea = 0
    total_case = 0
    joint_acc_a = 0
    total_case_a = 0
    total_case_e = 0
    num_batches = 0
    global_step = 0
    tp_a = 0
    fp_a = 0
    fn_a = 0
    tp = 0
    fp = 0
    fn = 0
    labels = []
    predicts = []
    with torch.no_grad():
        for batch in val_dataloader:
            num_batches += 1
            global_step += 1

            # Transfer to gpu
            if torch.cuda.is_available():
                if cfg.only_one_model:
                    #logits = proposal_model(input_ids=batch["input"], attention_mask=batch["input_attention"], token_type_ids=batch["input_type"]).logits # ,labels=batch["label"]
                    #probs = F.softmax(logits, dim=1)
                    #accs = ((batch["label"]==0)==(probs[:,0]>threshold)).cpu().tolist()
                    #labels.extend((batch["label"]==0).cpu().tolist())
                    #predicts.extend((probs[:,0]>threshold).cpu().tolist())
                    #acc.extend(accs) 
                    for i in range(len(batch['cases'])): # batch
                        # get_proposal_prob
                        p_batch = collate_fn(batch['cases'][i])
                        for key, val in p_batch.items():
                            if type(p_batch[key]) is list:
                                continue
                            p_batch[key] = p_batch[key].to(cfg.device[-1])
                        if p_batch!={}:
                            p_logits = proposal_model(input_ids=p_batch["input"], attention_mask=p_batch["input_attention"], token_type_ids=p_batch["input_type"]).logits
                            probs = F.softmax(p_logits, dim=1)
                            accept_prob = probs[:,0].cpu().tolist() # 0 means coherency in bert pretraining
                        else:
                            accept_prob = []
                        #Threshold
                        triple_num = len(accept_prob)
                        triples_accepted = []
                        triples_accepted_idx = []
                        accepted_probs = 1.0
                        triples = []
                        triple_probs = []
                        triple_idxs = []
                        proposals = [] # use proposals to indicate the possible results
                        org_proposals = []
                        gt = []
                        accept_result = []

                        # change propose to topk
                        for num in range(triple_num):
                            if batch['cases'][i][num]['label'] == 1:
                                gt.append(num)
                            if accept_prob[num] > origin_threshold :
                                accept_result.append(num)
                            if accept_prob[num]> accept_threshold:
                                triples_accepted.append(tokenizer.decode(batch['cases'][i][num]['triple']).replace('[CLS]','').replace('[SEP]','')) 
                                accepted_probs = accepted_probs*accept_prob[num] 
                                triples_accepted_idx.append(num)
                            elif accept_prob[num]> reject_threshold:
                                triples.append(tokenizer.decode(batch['cases'][i][num]['triple']).replace('[CLS]','').replace('[SEP]',''))
                                triple_probs.append(accept_prob[num])
                                triple_idxs.append(num)
                        proposals = [(triples_accepted, accepted_probs, triples_accepted_idx)]

                        # topk_num, get_topk here by using beam search, also somewhat like viterbi algorithm
                        for t_num in range(len(triples)):
                            triple = triples[t_num]
                            triple_prob = triple_probs[t_num]
                            triple_idx = triple_idxs[t_num]
                            new_proposals = [] # a temp variable to store the iterated proposal
                            for proposal in proposals:
                                new_proposals.append((proposal[0], proposal[1]*(1-triple_prob), proposal[2]))
                                tmp = copy.deepcopy(proposal)
                                tmp[0].append(triple)
                                #tmp[1] = proposal[1]*triple_prob
                                tmp[2].append(triple_idx)
                                new_proposals.append((tmp[0], proposal[1]*triple_prob, tmp[2]))
                            if len(new_proposals)>topk_num:
                                new_proposals.sort(key=lambda x:x[1], reverse=True)
                                proposals = copy.deepcopy(new_proposals[:topk_num])
                            else:
                                proposals = copy.deepcopy(new_proposals)
                        proposals.sort(key=lambda x:x[1], reverse=True)
                        topk = proposals[:topk_num]
                        #result = sorted(data,key=lambda x:(x[0],x[1].lower()))
                        # proposals.append('；'.join(proposal))
                        # proposed_probs.append(math.log(proposed_prob))
                        #org_proposals.append(org_proposal)
                        """
                        proposed_probs = []
                        for sample_num in range(cfg.train_sample_num): # cfg.test_num
                            proposal = []
                            org_proposal = []
                            gt = []
                            accept_result = []
                            random.random()
                            proposed_prob = 1.0
                            for num in range(triple_num):
                                p = random.random()
                                if p < accept_prob[num] + rescore_threshold:
                                    proposal.append(tokenizer.decode(batch['cases'][i][num]['triple']).replace('[CLS]','').replace('[SEP]','')) 
                                    org_proposal.append(num) # num batch['cases'][i][num]
                                    proposed_prob = proposed_prob*accept_prob[num]
                                else:
                                    proposed_prob = proposed_prob*(1-accept_prob[num])
                                if batch['cases'][i][num]['label'] == 1:
                                    gt.append(num)
                                if accept_prob[num] >(0.5 - rescore_threshold) :
                                    accept_result.append(num)
                                    # can directly concatenate all the triples to improve efficiency
                            proposals.append('；'.join(proposal))
                            proposed_probs.append(math.log(proposed_prob))
                            org_proposals.append(org_proposal)
                        """
                        #if dataset=='seretod':
                        to_be_reranked = ['；'.join(item[0]) for item in topk]
                        #else:
                        #    to_be_reranked = [' '.join(item[0]) for item in topk]
                        input = get_retrieval_sequence(tokenizer, [batch['context'][i]]*len(topk), to_be_reranked)
                        input.to(cfg.device[0])
                        #if cfg.add_extra_feature:
                        positive_count = torch.tensor([float(len(item[0])) for item in topk], dtype=torch.float).to(cfg.device[0])
                        if cfg.add_extra_feature:
                            logits = model(input_ids=input['input_ids'], attention_mask=input["attention_mask"], feature=positive_count).to('cpu').tolist()
                        else:
                            logits = model(input_ids=input['input_ids'], attention_mask=input["attention_mask"]).to('cpu').tolist()
                        if cfg.residual:
                            for j in range(len(logits)):
                                logits[j] = logits[j][0] + math.log(topk[j][1])
                        final = logits.index(max(logits))
                        triples = proposals[final][2]

                        # compute acc
                        total_case_e = total_case_e + 1
                        if gt!=[] and accept_result!=[]:
                            total_case_a = total_case_a + 1
                            joint_acc_a += set(accept_result)==set(gt)
                            joint_acc_ea += set(accept_result)==set(gt)
                        else:
                            joint_acc_ea += 1
                        for num in range(triple_num):
                            tp_a += (num in accept_result) and (num in gt)
                            fp_a += (num in accept_result) and (num not in gt)
                            fn_a += (num not in accept_result) and (num in gt)
                        if gt!=[] and triples!=[]:
                            total_case = total_case + 1
                            joint_acc += set(triples)==set(gt)
                            joint_acc_e += set(triples)==set(gt)
                        else:
                            joint_acc_e += 1
                        for num in range(triple_num):
                            tp += (num in triples) and (num in gt)
                            fp += (num in triples) and (num not in gt)
                            fn += (num not in triples) and (num in gt)
                #else:
                #    _,batch_acc,predict,label = model(input_sent=batch["context"], attention_sent=batch["context_attention"],input_triple=batch["triple"], attention_triple=batch["triple_attention"],label=batch["label"])
                #    labels.extend(label)
                #    predicts.extend(predict)
                #    acc.append(batch_acc)
    # the positive and negative are assigned to 0 and 1 respectively, so we should calculate the f1 of 0, dealt with by getting the probs of 0
    recall = tp/(tp+fn)
    precision= tp/(tp+fp)
    f1 = 2*precision*recall/(precision + recall+0.000001)
    recall_a = tp_a/(tp_a+fn_a)
    precision_a= tp_a/(tp_a+fp_a)
    f1_a = 2*precision_a*recall_a/(precision_a + recall_a+0.000001)
    # logging.info
    print("j_acc with empty slot: energy {}, proposal: {}".format(joint_acc_e/total_case_e, joint_acc_ea/total_case_e))
    print("proposal result: score: {}, precision: {}, recall: {}, f1: {}".format(joint_acc_a/total_case_a, precision_a, recall_a, f1_a))
    return joint_acc/total_case, precision, recall, f1

# getting retrieval results for baseline retrievers
def evaluate(model, val_dataloader, hparams): # tokenizer
    threshold = 0.1
    model.eval()
    acc = []
    num_batches = 0
    global_step = 0
    labels = []
    predicts = []
    with torch.no_grad():
        for batch in val_dataloader:
            num_batches += 1
            global_step += 1

            # Transfer to gpu
            if torch.cuda.is_available():
                for key, val in batch.items():
                    if type(batch[key]) is list:
                        continue
                    batch[key] = batch[key].to(hparams.device[0])
                if cfg.only_one_model:
                    logits = model(input_ids=batch["input"], attention_mask=batch["input_attention"], token_type_ids=batch["input_type"]).logits # ,labels=batch["label"]
                    probs = F.softmax(logits, dim=1)
                    accs = ((batch["label"]==0)==(probs[:,0]>threshold)).cpu().tolist()
                    labels.extend((batch["label"]==0).cpu().tolist())
                    predicts.extend((probs[:,0]>threshold).cpu().tolist())
                    acc.extend(accs) 
                else:
                    _,batch_acc,predict,label = model(input_sent=batch["context"], attention_sent=batch["context_attention"],input_triple=batch["triple"], attention_triple=batch["triple_attention"],label=batch["label"])
                    labels.extend(label)
                    predicts.extend(predict)
                    acc.append(batch_acc)
    # the positive and negative are assigned to 0 and 1 respectively, so we should calculate the f1 of 0
    positive = sum(labels)
    retrieved = sum((predicts[i] and labels[i]) for i in range(len(predicts)))
    predicted = sum(predicts)
    recall = retrieved/positive
    precision= retrieved/predicted
    f1 = 2*precision*recall/(precision + recall)
    return sum(acc)/len(acc), precision, recall, f1

def test(cfg, dataset):
    save_path = cfg.bert_save_path if dataset=='seretod' else (cfg.bert_save_path+dataset)
    tokenizer = BertTokenizer.from_pretrained(save_path)
    if cfg.only_one_model:
        model = BertForNextSentencePrediction.from_pretrained(save_path)#EncoderModel(cfg,tokenizer)
        model.to(cfg.device[0])
    else:
        model = EncoderModel(cfg,tokenizer)
        model.load_state_dict(torch.load(os.path.join(cfg.retrieval_save_path, "model.pt")))

    encoded_data = read_data(tokenizer, retrieve=True, dataset=dataset)
    test_dataloader=DataLoader(encoded_data['test'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn)
    acc, precision, recall, f1 = evaluate(model, test_dataloader, cfg)
    print("acc: {}, precision: {}, recall: {}, f1: {}".format(acc, precision, recall, f1))
    return recall
