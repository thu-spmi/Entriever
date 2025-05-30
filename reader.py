
import json
import os
import logging
import random
import re
import copy

import numpy as np
import torch
from transformers import BertTokenizer

from config import global_config as cfg
from KB_query import query
from collections import OrderedDict
from metrics import extract_request_info
# End of entity-name-list/user/entity-name/user-intent/KB-result/system-intent/system-response
special_tokens=['[EOS_L]', '[EOS_U]', '[EOS_E]', '[EOS_UI]', '[EOS_K]', '[EOS_SI]', '[EOS_S]']


def convert_to_sequences(data, dial_ids=None,posterior=False, return_dict =False):
    sequences=[]
    dicts=[]
    special_turns =[]
    KB_count = 0
    KB_statistic = {i:0 for i in range(30)}
    KB_counts = []
    for dial in data:
        dial_id=dial['id']
        if dial_ids is not None and dial_id not in dial_ids:
            continue
        EN_list={}
        user_EN_list={}
        new_dial = {}
        new_dial['log'] = []
        KB=dial['KB']
        kb_series = serialize_kb(KB)
        spoken = ''
        for turn in dial['content']:
            turn_dict ={}
            pv_EN_list=copy.deepcopy(user_EN_list)
            EN=set([])
            KB_result=[]
            KB_result_new=[]
            KB_result_gt=[]
            KB_result_gt_fg={}

            #get gtdb
            spoken = spoken + turn['用户']+ turn['客服']
            if 'info' in turn:
                for ent in turn['info']['ents']:
                    if ent['name'] in turn['用户'] and ent['name'].strip()!='NA': # this is an entity name mentioned by the user in current turn
                        if EN:
                            EN.add(ent['name'])
                        else:
                            EN = set([ent['name']])
                        ent_id=ent['id']
                        if ent_id not in EN_list:
                            EN_list[ent_id]=set([ent['name']])
                            user_EN_list[ent_id]=set([ent['name']])
                        else:
                            EN_list[ent_id].add(ent['name'])
                            if ent_id not in user_EN_list:
                                user_EN_list[ent_id]=set([ent['name']])
                            else:
                                user_EN_list[ent_id].add(ent['name'])
                
                # old version of adding gt_kb, not used because of the retrieval method
                for triple in turn['info']['triples']: 
                    if triple['ent-name'] in turn['用户'] and triple['ent-name'].strip()!='NA' and triple['ent-id']!='NA':
                        # this is an triple with ent-name and ent-id mentioned by the user in current turn
                        if EN:
                            EN.add(triple['ent-name'])
                        else:
                            EN = set([triple['ent-name']])
                        ent_id=triple['ent-id']
                        if ent_id not in EN_list:
                            EN_list[ent_id]=set([triple['ent-name']])
                            user_EN_list[ent_id]=set([triple['ent-name']])
                        else:
                            EN_list[ent_id].add(triple['ent-name'])
                            if ent_id not in user_EN_list:
                                user_EN_list[ent_id]=set([triple['ent-name']])
                            else:
                                user_EN_list[ent_id].add(triple['ent-name'])
                    if triple['value'] in turn['客服']: #and triple['value'] not in ','.join(KB_result_gt):
                        if triple['ent-name'].strip() =='NA' or triple['ent-name'] in ','.join(KB_result_gt):
                            KB_result_gt.append(triple['prop'] + ',' + triple['value'])
                        else:  #triple['ent-name'] not in KB_result_gt or triple['prop'] not in KB_result_gt:
                            KB_result_gt.append(triple['ent-name'] + ',' + triple['prop'] + ',' + triple['value'])
                 
            if KB_result !=[] and KB_result !='':
                KB_count = KB_count + 1 # 53263
            pv_EN_seq=','.join([','.join(list(item)) for item in list(pv_EN_list.values())])
            si=turn['客服意图']
            ui=turn['用户意图']
            si=re.sub(r'\(.*\)', '', si)
            ui=re.sub(r'ent-[0-9]+-', '', ui)
            ui=re.sub(r'ent--', '', ui)
            if EN:
                turn_ent =list(EN)
            else:
                turn_ent =[]
            if posterior:
                if cfg.act_change:
                    sequence=pv_EN_seq.lower()+'[EOS_L]'+turn['用户'].lower()+'[EOS_U]'+turn['客服'].lower()+'[EOS_S]'\
                        +','.join(turn_ent).lower()+'[EOS_E]'+ui.lower()+'[EOS_UI]'+ si.lower()+'[EOS_SI]' + ','.join(KB_result).lower()+'[EOS_K]'\
                        
                else:
                    sequence=pv_EN_seq.lower()+'[EOS_L]'+turn['用户'].lower()+'[EOS_U]'+turn['客服'].lower()+'[EOS_S]'\
                        +','.join(turn_ent).lower()+'[EOS_E]'+ui.lower()+'[EOS_UI]'+','.join(KB_result).lower()+'[EOS_K]'\
                        +si.lower()+'[EOS_SI]'
            else:
                if cfg.act_change:
                    sequence=pv_EN_seq.lower()+'[EOS_L]'+turn['用户'].lower()+'[EOS_U]'\
                        +','.join(turn_ent).lower()+'[EOS_E]'+ui.lower()+'[EOS_UI]' + turn['客服'].lower()+'[EOS_S]' + ','.join(KB_result).lower()+'[EOS_K]'\
                        +si.lower()+'[EOS_SI]'
                else:
                    sequence=pv_EN_seq.lower()+'[EOS_L]'+turn['用户'].lower()+'[EOS_U]'\
                        +','.join(turn_ent).lower()+'[EOS_E]'+ui.lower()+'[EOS_UI]'+','.join(KB_result).lower()+'[EOS_K]'\
                        +si.lower()+'[EOS_SI]'+turn['客服'].lower()+'[EOS_S]'
            sequences.append(sequence)
            if return_dict:
                turn_dict['ent_list'] = pv_EN_seq.lower()# note that eos_l are placed before user to simplify the code
                turn_dict['user'] = '[EOS_L]' + turn['用户'].lower()+'[EOS_U]'
                turn_dict['entity'] = ','.join(turn_ent).lower()+'[EOS_E]'
                turn_dict['bspn'] = ui.lower()+'[EOS_UI]'
                turn_dict['db'] = ','.join(KB_result).lower()+'[EOS_K]'
                turn_dict['db_new'] = ','.join(KB_result_new).lower()+'[EOS_K]'
                turn_dict['db_all'] = kb_series.lower()+'[EOS_K]'
                turn_dict['db_gt'] = '；'.join(KB_result_gt).lower()+'[EOS_K]' # ','.join(KB_result_gt).lower()
                turn_dict['aspn'] = si.lower()+'[EOS_SI]'
                turn_dict['resp'] = turn['客服'].lower()+'[EOS_S]'
                new_dial['log'].append(turn_dict)
                tmp = ''
                for s,v in KB_result_gt_fg.items():
                    tmp = tmp + s + ':' + ','.join(v) + '；'
                if '；' in tmp:
                    tmp = tmp[:-1]
                turn_dict['db_gtfg'] = tmp.lower()+'[EOS_K]'

                if len(KB_result_gt) not in KB_statistic:
                    KB_statistic[len(KB_result_gt)] = 0
                KB_statistic[len(KB_result_gt)] += 1
                KB_counts.append(len(KB_result_gt))

        if return_dict:
            dicts.append(new_dial)

    from matplotlib import pyplot as plt
    plt.hist(KB_counts, bins=[i+1 for i in range(10)]) #, bins=1)
    #plt.hist(KB_counts, bins=1)    
    plt.savefig('analysis.jpg')    
    print(KB_statistic)
    if return_dict:
        #kb_bad_path = 'data/extra/bad_kb.json'
        #json.dump(special_turns, open(kb_bad_path, 'w'),indent=2,ensure_ascii=False)
        return dicts
    else:
        return sequences

def serialize_kb(KB):
    kb_seq = []
    for e in KB:
        if e == 'NA':
            NA_temp = []
            for prop in KB['NA']:
                NA_temp.append(prop+':'+KB['NA'][prop])
            kb_seq.append(';'.join(NA_temp))
        else:
            ent_info = KB[e]
            ent_temp = []
            for ent in ent_info:
                if ent == 'name':
                    ent_temp.append('名称:'+ent_info[ent])
                elif ent == 'type':
                    ent_temp.append('类型:'+ent_info[ent])
                else:
                    ent_temp.append(ent+':'+ent_info[ent])
            kb_seq.append(';'.join(ent_temp))
    kb_seq = ';'.join(kb_seq)
    return kb_seq

def get_dial_hist(hist, turn, en=False):
    max_len = cfg.retrieve_hist
    new_hist = []
    hist_seq = ''
    if len(hist)>=max_len:
        for i in range(len(hist)-1):
            temp = hist[i]
            new_hist.append(hist[i+1])
            if en:
                hist_seq = ' user: ' + temp ['user'] + ' system: ' + temp['response']
            else:
                hist_seq = ' 用户：' + temp ['用户'] + ' 客服：' + temp['客服']
        new_hist.append(turn)
        if en:
            hist_seq = hist_seq + ' user: ' + turn['user']
        else:
            hist_seq = hist_seq + ' 用户：' + turn['用户']
    else:
        for temp in hist:
            if en:
                hist_seq = ' user: ' + temp ['user'] + ' system: ' + temp['response']
            else:
                hist_seq = ' 用户：' + temp ['用户'] + ' 客服：' + temp['客服']
        new_hist = hist
        new_hist.append(turn)
        if en:
            hist_seq = hist_seq + ' user: ' + turn['user']
        else:
            hist_seq = hist_seq + ' 用户：' + turn['用户']
    return new_hist, hist_seq

def get_pseudo_retrieval_data(tokenizer):
    # gpt_tokenizer is loaded for decoding previous pseudo label
    gpt_tokenizer = BertTokenizer.from_pretrained('experiments/baseline_post_gtdb/best_post_model')

    pseudo_data = json.load(open(cfg.pseudo_path,'r'))
    new_data = []
    special = []
    for dial in pseudo_data:
        new_dial = {'KB':[], 'content':[]}
        for turn in dial:
            new_turn = {}
            new_turn['用户'] = gpt_tokenizer.decode(turn['user'][1:-1]).replace(' ','')
            new_turn['客服'] = gpt_tokenizer.decode(turn['resp'][:-1]).replace(' ','')

            turn_kb = gpt_tokenizer.decode(turn['db_gt'][:-1]).replace(' ','').split('；')    
            turn_kb = list(set(turn_kb)) # remove repetition
            for item in turn_kb:
                flag = 0
                if ':' not in item:
                    turn_kb.remove(item)
                else:
                    values = item.split(':')[-1]
                    value = values.split(',') if (',' in values) else [values]
                    for v in value:
                        if v in new_turn['客服']:
                            flag = 1
                    if flag == 0:
                        special.append((new_turn['客服'], item))
                        turn_kb.remove(item)
            new_turn['kb'] = turn_kb
            new_dial['content'].append(new_turn)
            new_dial['KB'].extend(turn_kb)
        new_data.append(new_dial)
    cases=[]
    for dial in new_data:
        hist = []
        KB=dial['KB']
        for turn in dial['content']:
            positive = [] # kv_pairs
            negative = [] # kv_pairs
           
            # construct positive and negative sample from KB
            for case in turn['kb']:
                positive.append(tokenizer.encode(case))
            for item in KB:
                flag = 0
                values = item.split(':')[-1]
                value = values.split(',') if (',' in values) else [values]
                for v in value:
                    if v in turn['客服']:
                        flag = 1
                if flag == 0:
                    negative.append(tokenizer.encode(item))
            hist, hist_seq = get_dial_hist(hist, turn)
            seq = tokenizer.encode(hist_seq)
            for p in positive:
                cases.append({'context':seq,'triple':p,'label':1})
            random.shuffle(negative)
            SAMPLENUM = int(len(negative)*0.3)
            for n in negative[:SAMPLENUM]:
                cases.append({'context':seq,'triple':n,'label':-1})
    print(len(positive), len(negative))
    return cases

def get_retrieval_data(data, tokenizer, dial_ids=None, ebm=False):
    cases=[]
    max_len = 5
    for dial in data:
        #new_dial = []
        hist = []
        dial_id=dial['id']
        if dial_ids is not None and dial_id not in dial_ids:
            continue
        KB=dial['KB']
        spoken = ''
        for turn in dial['content']:
            turn_cases = []
            # note that spoken also includes the content in this turn
            spoken = spoken + turn['用户']+ turn['客服']
            positive = [] # kv_pairs
            negative = [] # kv_pairs
            #if 'info' in turn:
            #    for ent in turn['info']['ents']:
            #    for triple in turn['info']['triples']: 
            # construct positive and negative sample from KB
            for _,kvs in KB.items():
                for k,v in  kvs.items():
                    if k !='type': # remove type for now
                        if (',' not in v) and (v!= ''):
                            if v not in spoken:
                                negative.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ v))
                            elif v in turn['客服']:
                                positive.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ v))
                        else:
                            tmp_v = []
                            flag = 0
                            if cfg.fine_grained:
                                for value in v.split(','):
                                    if value not in spoken:
                                        negative.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ value))
                                    elif value in turn['客服']:
                                        positive.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ value))
                            else:
                                for value in v.split(','):
                                    if value not in spoken:
                                        tmp_v.append(value)
                                    elif value in turn['客服']:
                                        tmp_v.append(value)
                                        flag = 1
                                if (flag == 0) and (tmp_v!=[]):
                                    negative.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ ','.join(tmp_v)))   
                                elif flag == 1:
                                    positive.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ ','.join(tmp_v)))
            hist, hist_seq = get_dial_hist(hist, turn)
            seq = tokenizer.encode(hist_seq)
            #cases.append({'context':tokenizer.encode(hist_seq), 'positive':positive, 'negative':negative}) # change case to hist, triple, label
            for p in positive:
                #if cfg.only_one_model:
                #    cases.append({'context':hist_seq,'triple':p,'label':1})
                #else:
                turn_cases.append({'context':seq,'triple':p,'label':1})
            for n in negative:
                turn_cases.append({'context':seq,'triple':n,'label':-1})
            if not ebm:
                cases.extend(turn_cases)
            else:
                if turn_cases!=[]:
                    turn_case = {'context':hist_seq, 'cases':turn_cases}
                    cases.append(turn_case)

    return cases

def get_retrieval_data_en(data, tokenizer, dataset, ebm=False, split='train'):
    cases=[]
    max_len = 5
    for id, dial in data.items(): # in camrest and woz2.1, this means a turn
        dial_id = split + id
        if dataset=='incar':
            hist = []
            org_KB = dial['kg']
            KB = []
            for turn in dial['utterances']:
                entities = turn['reference_entities']
                for triple in org_KB:
                    kvs = ' '.join(triple)
                    flag = 0
                    for ent in entities:
                        if ent in kvs:
                            flag = 1
                            break
                    if flag==1:
                        if triple not in KB:
                            KB.append(triple)
            spoken = ''
            if KB!=[]:
                # head_entities = [k[0] for k in KB]
                for turn in dial['utterances']:
                    turn_cases = []
                    # note that spoken also includes the content in this turn
                    spoken = spoken + turn['user']+ turn['response']
                    positive = [] # kv_pairs
                    negative = [] # kv_pairs
                    targets = turn['reference_entities'] # list
                    all_kvs = []
                    for triples in KB:
                        kvs = ' '.join(triples)
                        target_kvs = ' '.join(triples[1:])
                        if kvs not in all_kvs:
                            all_kvs.append(kvs)
                            flag = 0
                            for t in targets:
                                if t in target_kvs: # target_kvs
                                    flag = 1
                                    break
                            if flag==0:
                                negative.append(tokenizer.encode(kvs.replace('_', ' ')))
                            else:
                                positive.append(tokenizer.encode(kvs.replace('_', ' ')))
                                
                    hist, hist_seq = get_dial_hist(hist, turn, en=True)
                    seq = tokenizer.encode(hist_seq)
                    for p in positive:
                        turn_cases.append({'context':seq,'triple':p,'label':1})
                    for n in negative:
                        turn_cases.append({'context':seq,'triple':n,'label':-1})
                    if not ebm:
                        cases.extend(turn_cases)
                    else:
                        if turn_cases!=[]:
                            turn_case = {'context':hist_seq, 'cases':turn_cases}
                            cases.append(turn_case)

        else: # in camrest and woz2.1
            if dataset=='woz2.1':
                org_KB = dial['kg']
            else:
                if dial['kg']!=[]:
                    org_KB = dial['kg']
                else:
                    if dial['reference_entities']!=[]:
                        if 'kg_tripe' in dial:
                            org_KB = dial['kg_tripe']
                        else:
                            org_KB = []
            hist_list = dial['history']
            if hist_list == []:
                KB = []
                cur_num = int(id)
                while(1):
                    cur_num += 1
                    if str(cur_num) in data:
                        kb_turn = data[str(cur_num)]
                        if kb_turn['history']==[]:
                            break
                        else:
                            entities = kb_turn['reference_entities']
                            for triple in org_KB:
                                kvs = ' '.join(triple)
                                flag = 0
                                for ent in entities:
                                    if ent in kvs:
                                        flag = 1
                                        break
                                if flag==1:
                                    if triple not in KB:
                                        KB.append(triple) 
                    else:
                        break
                
            turn_num = int(len(hist_list)/2)
            hist = [{'user':hist_list[2*num],'response':hist_list[2*num + 1]} for num in range(turn_num)]
            if KB!=[]:
                turn = dial
                turn_cases = []
                positive = [] # kv_pairs
                negative = [] # kv_pairs
                targets = turn['reference_entities'] # list
                for triples in KB:
                    kvs = ' '.join(triples)
                    target_kvs = ' '.join(triples[1:])
                    flag = 0
                    for t in targets: # 
                        if t in target_kvs: # kvs
                            flag = 1
                            break
                    if flag==0:
                        negative.append(tokenizer.encode(kvs.replace('_', ' ')))
                    else:
                        positive.append(tokenizer.encode(kvs.replace('_', ' ')))
                            
                hist, hist_seq = get_dial_hist(hist, turn, en=True)
                seq = tokenizer.encode(hist_seq)
                for p in positive:
                    turn_cases.append({'context':seq,'triple':p,'label':1})
                for n in negative:
                    turn_cases.append({'context':seq,'triple':n,'label':-1})
                if not ebm:
                    cases.extend(turn_cases)
                else:
                    if turn_cases!=[]:
                        turn_case = {'context':hist_seq, 'cases':turn_cases}
                        cases.append(turn_case)

    return cases

def get_retrieval_sequence(tokenizer, seqs, triples):
    final = [(seqs[s] + '[SEP]' + triples[s]).replace('[UNK]','g') for s in range(len(seqs))]
    tokenized = tokenizer(final, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return tokenized

def convert_to_test_sequences(data):
    dicts=[]
    for dial in data:
        new_dial = {}
        new_dial['KB'] = dial['KB']
        db_series = serialize_kb(dial['KB'])
        new_dial['org_goal'] = dial['goal']
        new_dial['goal'] = extract_request_info(dial['goal'],dial['KB'])
        new_dial['log'] = []
        spoken = ''
        for turn in dial['content']:
            KB_result_gt = []
            KB_result_gt_fg={}
            spoken = spoken + turn['用户']+ turn['客服']
            for _,kvs in dial['KB'].items():
                for k,v in  kvs.items():
                    if k !='type': # remove type for now
                        k_reg = k.replace('name','名称').replace('type','类型')
                        if (',' not in v) and (v!= ''):
                            if v in turn['客服']:
                                KB_result_gt.append(k_reg +':'+ v)
                                if k_reg not in KB_result_gt_fg:
                                    KB_result_gt_fg[k_reg] = []
                                if v not in KB_result_gt_fg[k_reg]:
                                    KB_result_gt_fg[k_reg].append(v)
                        else:
                            tmp_v = []
                            flag = 0
                            for value in v.split(','):
                                if value not in spoken:
                                    tmp_v.append(value)
                                elif value in turn['客服']:
                                    tmp_v.append(value)
                                    if k_reg not in KB_result_gt_fg:
                                        KB_result_gt_fg[k_reg] = []
                                    if value not in KB_result_gt_fg[k_reg]:
                                        KB_result_gt_fg[k_reg].append(value)
                                    flag = 1  
                            if flag == 1:
                                KB_result_gt.append(k.replace('name','名称').replace('type','类型') +':'+ ','.join(tmp_v))
            si=turn['客服意图']
            ui = turn['用户意图']
            si=re.sub(r'\(.*\)', '', si)
            ui=re.sub(r'ent-[0-9]+-', '', ui)
            ui=re.sub(r'ent--', '', ui)
            turn_dict={}
            turn_dict['user'] = '[EOS_L]' + turn['用户'].lower()+'[EOS_U]'
            turn_dict['bspn'] = ui.lower()+'[EOS_UI]'
            turn_dict['aspn'] = si.lower()+'[EOS_SI]'
            turn_dict['resp'] = turn['客服'].lower()+'[EOS_S]'
            turn_dict['db_gt'] = '；'.join(KB_result_gt).lower()+'[EOS_K]' # ','.join(KB_result_gt).lower()
            tmp = ''
            for s,v in KB_result_gt_fg.items():
                tmp = tmp + s + ':' + ','.join(v) + '；'
            if '；' in tmp:
                tmp = tmp[:-1]
            turn_dict['db_gtfg'] = tmp.lower()+'[EOS_K]'
            turn_dict['db_all'] = db_series.lower()+'[EOS_K]'
            new_dial['log'].append(turn_dict)

        dicts.append(new_dial)
    return dicts

def read_data(tokenizer, posterior = False, return_dict =False, retrieve=False, ebm=False, dataset='seretod'):
    if posterior:
        encoded_path=os.path.join(cfg.data_dir, 'encoded_data_post.json')
    else:
        encoded_path=os.path.join(cfg.data_dir, 'encoded_data.json')
    if return_dict:
        encoded_path=os.path.join(cfg.data_dir, 'encoded_data_dict.json')
    if retrieve:
        if cfg.fine_grained:
            encoded_path=os.path.join(cfg.data_dir, 'encoded_data_retrieve_fg.json')
        else:
            if dataset!='seretod':
                encoded_path=os.path.join(cfg.data_dir, f"encoded_data_retrieve_{dataset}_{cfg.retrieve_hist}_{cfg.pseudo_label_retrieval}.json")
            else:
                encoded_path=os.path.join(cfg.data_dir, f"encoded_data_retrieve_{cfg.retrieve_hist}_{cfg.pseudo_label_retrieval}.json")
    if ebm:
        if dataset!='seretod':
            data_path = f"encoded_data_ebm_{dataset}.json"
        else:
            data_path = f"encoded_data_ebm.json"
        encoded_path=os.path.join(cfg.data_dir, data_path)
    if not cfg.gpt:
        encoded_path = encoded_path.replace('.json','_t5.json')
    if not os.path.exists(encoded_path):
        """
        data=json.load(open(cfg.data_path, 'r', encoding='utf-8'))
        dial_ids=[dial['id'] for dial in data]
        random.shuffle(dial_ids)
        piece=len(dial_ids)//10
        """
        if dataset!='seretod':
            train_path = f"data_e2e/{dataset}/train.json"
            dev_path = f"data_e2e/{dataset}/val.json"
            test_path = f"data_e2e/{dataset}/val.json"
            train_data = json.load(open(train_path, 'r', encoding='utf-8'))
            dev_data = json.load(open(dev_path, 'r', encoding='utf-8'))
            test_data = json.load(open(test_path, 'r', encoding='utf-8'))
        else:
            dev_path = 'data/seretod/processed_data_dev.json'
            test_path = 'data/seretod/test_data_for_track2.json'
            train_data = json.load(open(cfg.data_path, 'r', encoding='utf-8'))
            dev_data = json.load(open(dev_path, 'r', encoding='utf-8'))
            test_data = json.load(open(test_path, 'r', encoding='utf-8'))
            train_ids = [dial['id'] for dial in train_data]
            dev_ids = [dial['id'] for dial in dev_data]
        # get seq data
        logging.info('Encoding data ...')
        if retrieve:
            encoded_data = {}
            encoded_data['train'] = []
            if cfg.pseudo_label_retrieval:
                encoded_data['train'].extend(get_pseudo_retrieval_data(tokenizer))
            if dataset!='seretod':
                encoded_data['train'].extend(get_retrieval_data_en(train_data, tokenizer, dataset=dataset, split='train'))
                encoded_data['dev'] = get_retrieval_data_en(dev_data, tokenizer, dataset=dataset, split='dev')
                encoded_data['test'] = get_retrieval_data_en(test_data, tokenizer, dataset=dataset, split='test')
            else:
                encoded_data['train'].extend(get_retrieval_data(train_data, tokenizer))
                encoded_data['dev'] = get_retrieval_data(dev_data, tokenizer)
                encoded_data['test'] = get_retrieval_data(test_data, tokenizer)
        elif ebm:
            encoded_data = {}
            encoded_data['train'] = []
            #if cfg.pseudo_label_retrieval:
            #    encoded_data['train'].extend(get_pseudo_retrieval_data(tokenizer))
            if dataset!='seretod':
                encoded_data['train'].extend(get_retrieval_data_en(train_data, tokenizer, dataset=dataset, ebm=True, split='train'))
                encoded_data['dev'] = get_retrieval_data_en(dev_data, tokenizer, dataset=dataset, ebm=True, split='dev')
                encoded_data['test'] = get_retrieval_data_en(test_data, tokenizer, dataset=dataset, ebm=True, split='test')
            else:
                encoded_data['train'].extend(get_retrieval_data(train_data, tokenizer, ebm=True))
                encoded_data['dev'] = get_retrieval_data(dev_data, tokenizer, ebm=True)
                encoded_data['test'] = get_retrieval_data(test_data, tokenizer, ebm=True)
        else:
            train_seqs=convert_to_sequences(train_data, train_ids,posterior, return_dict = return_dict)
            dev_seqs=convert_to_sequences(dev_data, dev_ids,posterior, return_dict = return_dict)
            test_seqs=convert_to_test_sequences(test_data)
            logging.info('Dialogs -- Train:{}, dev:{}, test:{}'.format(len(train_ids), len(dev_ids), len(test_data)))
            logging.info('Sequences -- Train:{}, dev:{}, test:{}'.format(len(train_seqs), len(dev_seqs), len(test_seqs)))
            seq_data={
                'train':train_seqs,
                'dev':dev_seqs,
                'test':test_seqs
            }
            dial_id_data={
                'train':train_ids,
                'dev':dev_ids
            }
            json.dump(dial_id_data, open(os.path.join(cfg.data_dir, 'dial_ids.json'), 'w'))
            if not return_dict:
                json.dump(seq_data, open(os.path.join(cfg.data_dir, 'all_data.json'), 'w'),indent=2, ensure_ascii=False)
            else:
                encoded_data={}
                for s in ['train', 'dev', 'test']:
                    encoded_data[s]=[]
                    for seq in seq_data[s]:
                        if return_dict:
                            encoded_dial = []
                            if s =='train':
                                KB = seq['KB'] # add KB for retrieval in training
                                for turn in seq['log']:
                                    encoded_turn = {}
                                    for key,value in turn.items():
                                        encoded_turn[key] = tokenizer.encode(value)[1:-1]
                                    encoded_turn['KB'] = KB
                                    encoded_dial.append(encoded_turn)
                            else:
                                KB = seq['KB'] # KB for evaluation
                                goal = seq['goal']
                                org_goal = seq['org_goal']
                                for turn in seq['log']:
                                    encoded_turn = {}
                                    for key,value in turn.items():
                                        encoded_turn[key] = tokenizer.encode(value)[1:-1]
                                    encoded_turn['KB'] = KB
                                    encoded_turn['goal'] = goal
                                    encoded_turn['org_goal'] = org_goal
                                    encoded_dial.append(encoded_turn)
                            encoded_data[s].append(encoded_dial)
                        else:
                            if s!='test':
                                encoded_data[s].append(tokenizer.encode(seq))
        json.dump(encoded_data, open(encoded_path, 'w'))
        logging.info('Data encoded, saved in:{}'.format(encoded_path))
    else:
        logging.info('Reading encoded data from:{}'.format(encoded_path))
        encoded_data=json.load(open(encoded_path, 'r'))
        if cfg.with_pseudo_label :
            pseudo_data = json.load(open(cfg.pseudo_path, 'r'))
            random.shuffle(pseudo_data)
            pseudo_num = cfg.pseudo_porportion*len(encoded_data['train'])
            encoded_data['train'].extend(pseudo_data[:pseudo_num])
    logging.info('Train:{}, dev:{}, test:{}'.format(len(encoded_data['train']), len(encoded_data['dev']), len(encoded_data['test'])))
    return encoded_data

def convert_to_dict(tokenizer,data):
    new_data = []
    for dial in data:
        new_dial=[]
        for turn in dial:
            enc = {}
            if '[SPEAKER 2]' in turn:
                enc['user'] = tokenizer.encode('[EOS_L]' + turn['[SPEAKER 2]'].lower()+'[EOS_U]')[1:-1]
            elif '[SPEAKER <B>]' in turn:
                enc['user'] = tokenizer.encode('[EOS_L]' + turn['[SPEAKER <B>]'].lower()+'[EOS_U]')[1:-1]
            else:
                break
            if '[SPEAKER 1]' in turn:
                enc['resp'] = tokenizer.encode(turn['[SPEAKER 1]'].lower()+'[EOS_S]')[1:-1]
            elif '[SPEAKER <A>]' in turn:
                enc['resp'] = tokenizer.encode(turn['[SPEAKER <A>]'].lower()+'[EOS_S]')[1:-1]
            else:
                break
            if enc!={}:
                new_dial.append(enc)
        if new_dial!=[]:
            new_data.append(new_dial)
    return new_data

def get_unsup(tokenizer, dial_num = 0, pretrain = False):#,ratio
    unlab_num = cfg.jsa_unsup*dial_num  #controls the num of unlab dials, # 1, 2, 4, 9
    encoded_path=os.path.join(cfg.data_dir, 'encoded_data_unl_'+str(unlab_num)+'.json')
    if pretrain:
        encoded_path=os.path.join(cfg.data_dir, 'encoded_data_unl_whole.json') if cfg.gpt else os.path.join(cfg.data_dir, 'encoded_data_unl_whole_t5.json')
    if not os.path.exists(encoded_path):
        unl=json.load(open('data/data_unlabel_process.json', 'r', encoding='utf-8'))
        #total_num = turn_num * ratio//10   #average dialog length, can be adjusted
        random.shuffle(unl)
        if pretrain:
            data = unl
        else:
            data = unl[:unlab_num] #data with low ratio are contained in data with high ratio
        logging.info('Encoding data ...')
        encoded_data=convert_to_dict(tokenizer,data)        
        json.dump(encoded_data, open(encoded_path, 'w'))
        logging.info('Data encoded, saved in:{}'.format(encoded_path))
    else:
        logging.info('Reading encoded data from:{}'.format(encoded_path))
        encoded_data=json.load(open(encoded_path, 'r'))
    if pretrain:
        new_data = {}
        sequences = []
        for dial in encoded_data:
            for turn in dial:
                if cfg.gpt:
                    sequences.append([tokenizer.convert_tokens_to_ids('[CLS]')] + turn['user']+
                    turn['resp'] + [tokenizer.convert_tokens_to_ids('[SEP]')])#+ tokenizer.convert_tokens_to_ids(['[EOS_E]','[EOS_UI]','[EOS_K]','[EOS_SI]'])
                else:
                    sequences.append({'input':([tokenizer.convert_tokens_to_ids('[CLS]')] + turn['user']),
                    'output':(turn['resp'] + [tokenizer.convert_tokens_to_ids('[SEP]')])})
        piece=len(sequences)//10
        new_data['dev'] = sequences[9*piece:]
        new_data['train'] =sequences[:9*piece]
        return new_data
    else:
        return encoded_data

def extract_test_dial(data='test'):
    dial_ids=json.load(open(os.path.join(cfg.data_dir, 'dial_ids.json'), 'r', encoding='utf-8'))
    all_data=json.load(open(cfg.data_path, 'r', encoding='utf-8'))
    test_data=[]
    for dial in all_data:
        if dial['id'] in dial_ids[data]:
            test_data.append(dial)
    return test_data

def _bucket_by_turn(encoded_data):
    turn_bucket = {}
    for dial in encoded_data:
        turn_len = len(dial)
        #修改记录：对于turn_len=0情况的处理
        if turn_len==0:
            continue
        if turn_len not in turn_bucket:
            turn_bucket[turn_len] = []
        turn_bucket[turn_len].append(dial)
    del_l = []
    for k in turn_bucket:
        if k >= 5:
            del_l.append(k)
        logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
   
    return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

def _construct_mini_batch(data,batch_size):
    all_batches = []
    batch = []
    for dial in data:
        batch.append(dial)
        if len(batch) == batch_size:
            # print('batch size: %d, batch num +1'%(len(batch)))
            all_batches.append(batch)
            batch = []

    if len(batch)>0:
        all_batches.append(batch)
    return all_batches

def inverse_transpose_batch(turn_batch_list):
    """
    :param turn_batch_list: list of transpose dial batch
    """
    dialogs = []
    total_turn_num = len(turn_batch_list)
    # initialize
    for idx_in_batch, _ in enumerate(turn_batch_list[0]['user']):
        dialog = []
        for turn_n in range(total_turn_num):
            dial_turn = {}
            turn_batch = turn_batch_list[turn_n]
            for key, v_list in turn_batch.items():
                value = v_list[idx_in_batch]
                dial_turn[key] = value
            dialog.append(dial_turn)
        dialogs.append(dialog)
    return dialogs

def inverse_transpose_turn(turn_list):
    turn_num = len(turn_list)
    dialog = []
    for turn_idx in range(turn_num):
        dial_turn = {}
        turn = turn_list[turn_idx]
        for key, value in turn.items():
            if key=='dial_id':
                continue
            dial_turn[key] = value
        dialog.append(dial_turn)
    return dialog
    
def get_batches(dials,batch_size):
    # organize data by batches  
    
    turn_bucket = _bucket_by_turn(dials)
    all_batches = []
    num_training_steps = 0
    num_turns = 0
    num_dials = 0

    for k in turn_bucket:
        if k == 1 or k >= 17:  #max turn num 17
            continue
        batches = _construct_mini_batch(turn_bucket[k],batch_size)
        if len(batches)==0:
            continue
        #log_str += "turn num:%d, dial num: %d, batch num: %d last batch len: %d\n" % (
        #    k, len(turn_bucket[k]), len(batches), len(batches[-1]))
        
        num_training_steps += k * len(batches)
        num_turns += k * len(turn_bucket[k])
        num_dials += len(turn_bucket[k])
        all_batches += batches
    #log_str += 'total batch num: %d\n' % len(all_batches)    
    random.shuffle(all_batches)
    return all_batches

def split_turn_batch(turn_batch, batch_size, other_batch=None):
    batches=[]
    other_batches=[]
    B=len(turn_batch['user'])
    for i in range(0, B, batch_size):
        new_turn_batch={}
        if other_batch:
            other_batches.append(other_batch[i:i+batch_size])
        for key in turn_batch:
            new_turn_batch[key]=turn_batch[key][i:i+batch_size]
        batches.append(new_turn_batch)
    if other_batch:
        return batches, other_batches
    else:
        return batches, None

def transpose_batch(batch):
    dial_batch = []
    turn_num = len(batch[0]) 
    for turn in range(turn_num):
        turn_l = {}
        for dial in batch:
            this_turn = dial[turn]
            for k in this_turn:
                if k not in turn_l:
                    turn_l[k] = []
                turn_l[k].append(this_turn[k])
        dial_batch.append(turn_l)
    return dial_batch

def convert_batch_t5(cls, sep, turn_batch, pv_batch, first_turn=False, posterior=False):
    inputs = {}
    labels = {}
    contexts = []
    label_contexts = []
    if first_turn:   
        if cfg.gt_db:
            batch_zipped=zip(turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], (turn_batch['db_gtfg'] if cfg.fine_grained else turn_batch['db_gt']),turn_batch['aspn'], turn_batch['resp'])
        else:
            batch_zipped=zip(turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], turn_batch['db'],turn_batch['aspn'], turn_batch['resp'])
        if cfg.db_change:
            batch_zipped = zip(turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], turn_batch['db_all'], turn_batch['aspn'], turn_batch['resp'])    
        for u, b, ent, db, a, r in batch_zipped:
            if posterior:
                if cfg.no_user_intent:
                    context = [cls] + u + r # + [sep]
                    label_context = db + a + [sep] # [cls] +
                else:
                    context=[cls] + u + r # + [sep]#add [cls] and [sep] token
                    label_context= ent + b + db + a + [sep] 
            else:
                if cfg.no_user_intent:
                    context = ([cls] + db + u) if cfg.kb_grounding else ([cls] + u + db)# + [sep]
                    label_context = a + r + [sep] # [cls] +
                else: # not support as multi imput and multi output needed
                    context = [cls] + u + ent + b + db + a + r + [sep]
                    label_context=(len(u)+1)*[cfg.pad_id] + ent + b + len(db)*[cfg.pad_id] + a + r + [sep]  # db can be pad in priori model
            contexts.append(context)
            label_contexts.append(label_context)
    else:
        if cfg.gt_db:
            batch_zipped = zip(pv_batch,turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], (turn_batch['db_gtfg'] if cfg.fine_grained else turn_batch['db_gt']), turn_batch['aspn'], turn_batch['resp'])
        else:
            batch_zipped = zip(pv_batch,turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], turn_batch['db'], turn_batch['aspn'], turn_batch['resp'])
        #if cfg.db_change:
        #    batch_zipped = zip(pv_batch,turn_batch['user'], turn_batch['bspn'], 
        #        turn_batch['entity'], turn_batch['db_all'], turn_batch['aspn'], turn_batch['resp'])
        for ur, u, b, ent, db, a, r in batch_zipped:
            if posterior: # act_change not added
                if cfg.no_user_intent:
                    context = [cls] + u + r # + [sep] # [cls] + ur + u + r + db + a + [sep]
                    label_context = db + a + [sep] # [cls] +
                else:
                    context = [cls] + ur + u + r # + [sep]
                    label_context= ent + b + db + a + [sep]
            elif cfg.db_change:
                context = [cls] + db + ur  + u # + [sep]
                label_context = b + a + r + [sep]
            else:
                if cfg.no_user_intent:
                    context = ([cls] + db + u) if cfg.kb_grounding else ([cls] + u + db)
                    # + [sep] # [cls] + ur + u + db + a + r + [sep]
                    label_context = a + r + [sep] # ur [cls] +
                else: # not supported
                    context = [cls] + ur + u + ent + b + db + a + r + [sep]
                    label_context=(len(ur+u)+1)*[cfg.pad_id] + ent + b + len(db)*[cfg.pad_id] + a + r + [sep]
            contexts.append(context)
            label_contexts.append(label_context)
    inputs['contexts'] = contexts
    inputs['contexts_np'], inputs['attention'] = padSeqs_gpt(inputs['contexts'], cfg.pad_id, attention=True)
    labels['contexts'] = label_contexts
    labels['contexts_np'], labels['attention'] = padSeqs_gpt(labels['contexts'], cfg.pad_id, attention=True)
    return inputs, labels
    
def convert_batch_turn(cls, sep, turn_batch, pv_batch, first_turn=False, posterior=False):
    '''
    Args:
    Returns:
    '''
    inputs = {}
    labels = {}
    contexts = []
    label_contexts = []
    if first_turn:   
        if cfg.gt_db:
            batch_zipped=zip(turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], (turn_batch['db_gtfg'] if cfg.fine_grained else turn_batch['db_gt']),turn_batch['aspn'], turn_batch['resp'])
        elif cfg.joint_training:
            batch_zipped=zip(turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], turn_batch['db_retrieval'],turn_batch['aspn'], turn_batch['resp'])
        else:
            batch_zipped=zip(turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], turn_batch['db'],turn_batch['aspn'], turn_batch['resp'])
        if cfg.db_change:
            batch_zipped = zip(turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], turn_batch['db_all'], turn_batch['aspn'], turn_batch['resp'])    
        for u, b, ent, db, a, r in batch_zipped:
            if posterior:
                if cfg.posterior_change:
                    context=[cls] + u + ent + b + r + db + a + [sep]#add [cls] and [sep] token
                    label_context=(len(u)+1)*[cfg.pad_id] + ent + b + len(r)*[cfg.pad_id] + db + a + [sep] #len(r)*[cfg.pad_id]
                elif cfg.act_change:
                    context=[cls] + u + r + ent + b + a + db + [sep]
                    label_context=(len(u+r)+1)*[cfg.pad_id] + ent + b + a + db + [sep]
                elif cfg.no_user_intent:
                    context=[cls] + u + r + db + a + [sep]
                    label_context=(len(u+r)+1)*[cfg.pad_id] + db + a + [sep]
                else:
                    context=[cls] + u + r + ent + b + db + a + [sep]#add [cls] and [sep] token
                    label_context=(len(u+r)+1)*[cfg.pad_id] + ent + b + db + a + [sep] 
            elif cfg.db_change:
                context = [cls] + db + u + b + a + r + [sep]
                label_context=(len(u+db)+1)*[cfg.pad_id] + b + a + r + [sep]
            else:
                if cfg.act_change:
                    context = [cls] + u + ent + b + a + db + r + [sep]
                    label_context=(len(u)+1)*[cfg.pad_id] + ent + b + a + len(db)*[cfg.pad_id] + r + [sep]
                elif cfg.no_user_intent:
                    context = [cls] + ((db + u) if cfg.kb_grounding else (u + db)) + a + r + [sep]
                    label_context=(len(u)+1)*[cfg.pad_id] + len(db)*[cfg.pad_id] + a + r + [sep]
                else:
                    context = [cls] + u + ent + b + db + a + r + [sep]
                    label_context=(len(u)+1)*[cfg.pad_id] + ent + b + len(db)*[cfg.pad_id] + a + r + [sep]  # db can be pad in priori model
            contexts.append(context)
            label_contexts.append(label_context)
    else:
        if cfg.gt_db:
            batch_zipped = zip(pv_batch,turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], (turn_batch['db_gtfg'] if cfg.fine_grained else turn_batch['db_gt']), turn_batch['aspn'], turn_batch['resp'])
        elif cfg.joint_training:
            batch_zipped = zip(pv_batch,turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], (turn_batch['db_gtfg'] if cfg.fine_grained else turn_batch['db_retrieval']), turn_batch['aspn'], turn_batch['resp'])
        else:
            batch_zipped = zip(pv_batch,turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], turn_batch['db'], turn_batch['aspn'], turn_batch['resp'])
        if cfg.db_change:
            batch_zipped = zip(pv_batch,turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], turn_batch['db_all'], turn_batch['aspn'], turn_batch['resp'])
        for ur, u, b, ent, db, a, r in batch_zipped:
            if posterior: # act_change not added
                if cfg.posterior_change:
                    context = [cls] + ur + u + ent + b + r + db + a + [sep]
                    label_context=(len(ur+u)+1)*[cfg.pad_id] + ent + b + len(r)*[cfg.pad_id] + db + a + [sep]#len(r)*[cfg.pad_id]
                elif cfg.no_user_intent:
                    context = [cls] + (ur if cfg.with_context else []) + u + r + db + a + [sep] # [cls] + ur + u + r + db + a + [sep]
                    label_context=(len((ur if cfg.with_context else []) + u + r)+1)*[cfg.pad_id] + db + a + [sep]
                else:
                    context = [cls] + ur + u + r + ent + b + db + a + [sep]
                    label_context=(len(ur+u+r)+1)*[cfg.pad_id] + ent + b + db + a + [sep]
            elif cfg.db_change:
                context = [cls] + db + ur  + u + b + a + r + [sep]
                label_context=(len(u+db+ur)+1)*[cfg.pad_id] + b + a + r + [sep]
            else:
                if cfg.act_change:
                    context = [cls] + ur + u + ent + b + a + db + r + [sep]
                    label_context=(len(ur+u)+1)*[cfg.pad_id] + ent + b + a + len(db)*[cfg.pad_id] + r + [sep]
                elif cfg.no_user_intent:
                    context = [cls] + (ur if cfg.with_context else []) + ((db + u) if cfg.kb_grounding else (u + db)) + a + r + [sep] # [cls] + ur + u + db + a + r + [sep]
                    label_context=(len((ur if cfg.with_context else [])+u)+1)*[cfg.pad_id] + len(db)*[cfg.pad_id] + a + r + [sep] # ur
                else:
                    context = [cls] + ur + u + ent + b + db + a + r + [sep]
                    label_context=(len(ur+u)+1)*[cfg.pad_id] + ent + b + len(db)*[cfg.pad_id] + a + r + [sep]
            contexts.append(context)
            label_contexts.append(label_context)
    inputs['contexts'] = contexts
    inputs['contexts_np'], inputs['lengths'] = padSeqs_gpt(inputs['contexts'], cfg.pad_id)
    labels['contexts']=label_contexts
    labels['contexts_np'], labels['lengths']= padSeqs_gpt(labels['contexts'], cfg.pad_id)
    return inputs, labels

def convert_eval_batch_turn(cls, turn_batch, pv_batch, mode='gen_bspn', bspn_gen=None, ent_gen=None, db_gen = None, a_gen = None, posterior=False):
    eval_batch=[]
    assert mode in ['gen_ent','gen_bspn', 'gen_kb','gen_ar','gen_resp']
    if pv_batch is None:
        if mode=='gen_ent':
            for u, r in zip(turn_batch['user'], turn_batch['resp']):#only eos_ent_id is used
                context= [cls] + u + r if (posterior and (not cfg.posterior_change))  else [cls] + u
                eval_batch.append(context)
        if mode=='gen_bspn':
            if cfg.db_change:
                for u, r, db in zip(turn_batch['user'], turn_batch['resp'],db_gen):
                    context= [cls] + db + u + r if (posterior and (not cfg.posterior_change)) else [cls] + db + u 
                    eval_batch.append(context)
            else:
                for u, r, ent in zip(turn_batch['user'], turn_batch['resp'],ent_gen):
                    context= [cls] + u + r + ent if (posterior and (not cfg.posterior_change)) else [cls] + u + ent
                    eval_batch.append(context)
        if mode=='gen_kb':
            #if cfg.act_change:
            if cfg.no_user_intent:
                for u, r in zip(turn_batch['user'], turn_batch['resp']):
                    context= [cls] + u + r if posterior else [cls] + u
                    eval_batch.append(context)
            else:
                for u, r, ent, b in zip(turn_batch['user'], turn_batch['resp'],ent_gen,bspn_gen):
                    context= [cls] + u + r + ent + b if posterior else [cls] + u + ent + b
                    eval_batch.append(context)
            if (posterior and cfg.posterior_change):
                for u, r, ent, b in zip(turn_batch['user'], turn_batch['resp'],ent_gen,bspn_gen):
                    context= [cls] + u + ent + b + r
                    eval_batch.append(context)
        if mode=='gen_ar':
            if cfg.db_change:
                for u, r, b, db in zip(turn_batch['user'], turn_batch['resp'], bspn_gen, db_gen):
                    context= [cls] + db + u + r +  b if posterior else [cls] + db + u + b
                    eval_batch.append(context)
            else:
                if cfg.no_user_intent and (not (posterior and cfg.posterior_change)): # prior mode
                    for u, r, db in zip(turn_batch['user'], turn_batch['resp'], db_gen):
                        context= [cls] + u + r + db if posterior else [cls] + ((db + u) if cfg.kb_grounding else (u + db))
                        eval_batch.append(context)
                elif (posterior and cfg.posterior_change):
                    for u, r, ent, b, db in zip(turn_batch['user'], turn_batch['resp'], ent_gen, bspn_gen, db_gen):
                        context= [cls] + u + ent + b + r + db
                        eval_batch.append(context)
                else:
                    for u, r, ent, b, db in zip(turn_batch['user'], turn_batch['resp'], ent_gen, bspn_gen, db_gen):
                        context= [cls] + u + r + ent + b + db if posterior else [cls] + u + ent + b +db
                        eval_batch.append(context)
        if mode=='gen_resp':
            #if cfg.act_change: only happen in this mode
            for u, r, ent, b, db, a in zip(turn_batch['user'], turn_batch['resp'],ent_gen, bspn_gen, db_gen, a_gen):
                context= [cls] + u + r + ent + b if posterior else [cls] + u + ent + b + a + db
                eval_batch.append(context)
    else:
        if mode=='gen_ent':
            for hist, u, r in zip(pv_batch, turn_batch['user'], turn_batch['resp']):
                context = [cls] + hist + u + r  if (posterior and (not cfg.posterior_change)) else [cls] + hist + u 
                eval_batch.append(context)
        if mode=='gen_bspn':
            if cfg.db_change:
                for hist, u, r, db in zip(pv_batch, turn_batch['user'], turn_batch['resp'],db_gen):
                    context = [cls] + db + hist + u + r  if (posterior and (not cfg.posterior_change)) else [cls] + db + hist + u
                    eval_batch.append(context)
            else:
                for hist, u, r, ent in zip(pv_batch, turn_batch['user'], turn_batch['resp'],ent_gen):
                    context = [cls] + hist+u + r + ent if (posterior and (not cfg.posterior_change)) else [cls] + hist+u + ent
                    eval_batch.append(context)
        if mode=='gen_kb':
            if cfg.no_user_intent:
                for hist, u, r in zip(pv_batch, turn_batch['user'], turn_batch['resp']):
                    context = [cls] + (hist if cfg.with_context else []) + u + r if posterior else [cls] +(hist if cfg.with_context else []) +  u # hist + 
                    eval_batch.append(context) 
            else:
                for hist, u, r, ent, b in zip(pv_batch, turn_batch['user'], turn_batch['resp'],ent_gen,bspn_gen):
                    context = [cls] + hist + u +r  + ent + b if posterior else [cls] + hist + u + ent + b
                    eval_batch.append(context)
            if (posterior and cfg.posterior_change):
                for hist, u, r, ent, b in zip(pv_batch, turn_batch['user'], turn_batch['resp'],ent_gen,bspn_gen):
                    context= [cls] + hist + u + ent + b + r
                    eval_batch.append(context)
        if mode=='gen_ar':
            if cfg.db_change:
                for hist,u, r, b,db in zip(pv_batch, turn_batch['user'], turn_batch['resp'], bspn_gen, db_gen):
                    context= [cls] + db + hist + u + r +  b if posterior else [cls] + db + hist + u + b
                    eval_batch.append(context)
            else:
                if cfg.no_user_intent:
                    for hist, u, r, db in zip(pv_batch, turn_batch['user'], turn_batch['resp'], db_gen):
                        context= [cls] + (hist if cfg.with_context else []) + u + r + db if posterior else [cls] + (hist if cfg.with_context else []) + ((db + u) if cfg.kb_grounding else (u + db)) # hist+
                        eval_batch.append(context)
                else:
                    for hist, u, r, ent, b, db in zip(pv_batch, turn_batch['user'], turn_batch['resp'],ent_gen,bspn_gen, db_gen):
                        context = [cls] + hist+ u + r + ent + b + db if posterior else [cls] + hist+ u + ent + b + db
                        #eval_batch.append(context), not available now
                if (posterior and cfg.posterior_change):
                    for hist, u, r, ent, b, db in zip(pv_batch, turn_batch['user'], turn_batch['resp'],ent_gen,bspn_gen, db_gen):
                        context= [cls] + hist + u + ent + b + r + db
                        eval_batch.append(context)
        if mode=='gen_resp':
            #if cfg.act_change:
            for hist, u, r, ent, b, db, a in zip(pv_batch, turn_batch['user'], turn_batch['resp'],ent_gen, bspn_gen, db_gen, a_gen):
                context= [cls] + hist + u + ent + b + a + db
                eval_batch.append(context)
    return eval_batch

def convert_eval_batch_t5(cls, turn_batch, pv_batch, mode='gen_bspn', bspn_gen=None, ent_gen=None, db_gen = None, a_gen = None, posterior=False):
    eval_batch=[]
    assert mode in ['gen_kb','gen_ar','gen_resp']
    if pv_batch is None:
        if mode=='gen_kb':
            #if cfg.act_change:
            for u, r, ent, b in zip(turn_batch['user'], turn_batch['resp'],ent_gen,bspn_gen):
                if cfg.no_user_intent:
                    context= [cls] + u + r if posterior else [cls] + u
                else:
                    context= [cls] + u + r + ent + b if posterior else [cls] + u + ent + b
                if (posterior and cfg.posterior_change):
                    context= [cls] + u + ent + b + r
                eval_batch.append(context)
        if mode=='gen_ar':
            if cfg.db_change:
                for u, r, b,db in zip(turn_batch['user'], turn_batch['resp'], bspn_gen, db_gen):
                    context= [cls] + db + u + r +  b if posterior else [cls] + db + u + b
                    eval_batch.append(context)
            else:
                for u, r, ent, b,db in zip(turn_batch['user'], turn_batch['resp'], ent_gen, bspn_gen, db_gen):
                    if cfg.no_user_intent:
                        context= [cls] + u + r + db if posterior else [cls] + u +db
                    else:
                        context= [cls] + u + r + ent + b + db if posterior else [cls] + u + ent + b +db
                    if (posterior and cfg.posterior_change):
                        context= [cls] + u + ent + b + r + db
                    eval_batch.append(context)
    else:
        if mode=='gen_kb':
            for hist, u, r, ent, b in zip(pv_batch, turn_batch['user'], turn_batch['resp'],ent_gen,bspn_gen):
                if cfg.no_user_intent:
                    context = [cls] + u + r if posterior else [cls] + u # hist + 
                else:
                    context = [cls] + hist + u +r  + ent + b if posterior else [cls] + hist + u + ent + b
                if (posterior and cfg.posterior_change):
                    context= [cls] + hist + u + ent + b + r
                eval_batch.append(context)
        if mode=='gen_ar':
            if cfg.db_change:
                for hist,u, r, b,db in zip(pv_batch, turn_batch['user'], turn_batch['resp'], bspn_gen, db_gen):
                    context= [cls] + db + hist + u + r +  b if posterior else [cls] + db + hist + u + b
                    eval_batch.append(context)
            else:
                for hist, u, r, ent, b, db in zip(pv_batch, turn_batch['user'], turn_batch['resp'],ent_gen,bspn_gen, db_gen):
                    if cfg.no_user_intent:
                        context= [cls] + u + r + db if posterior else [cls] + u + db # hist+
                    else:
                        context = [cls] + hist+ u + r + ent + b + db if posterior else [cls] + hist+ u + ent + b + db
                    if (posterior and cfg.posterior_change):
                        context= [cls] + hist + u + ent + b + r + db
                    eval_batch.append(context)
    return eval_batch

def padSeqs_gpt(sequences, pad_id, maxlen=None, attention=False):
    lengths = []
    for x in sequences:
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_mexlen = np.max(lengths)
    # maxlen = 512
    if seq_mexlen > 512: 
        maxlen = 512
    else:
        maxlen = seq_mexlen
    
    x = (np.ones((num_samples, maxlen)) * pad_id)
    attentions = (np.ones((num_samples, maxlen)) * pad_id)
    for idx, s in enumerate(sequences):
        if not len(s):
            print('empty list was found in padSeqs')
        trunc = s[-maxlen:]
        trunc = np.asarray(trunc)
        x[idx, :len(trunc)] = trunc
        attentions[idx, :len(trunc)] = np.ones(len(trunc))
    if attention:
        return x, attentions
    else:  
        return x, lengths

def train_collate_fn(batch):
    if isinstance(batch[0],dict):
        batch_tensor = {}
        for key in batch[0]:
            input = [b[key] for b in batch]
            padded = padSeqs(input, cfg.pad_id)
            batch_tensor[key] = torch.from_numpy(padded).long()
    else:
        pad_batch = padSeqs(batch, cfg.pad_id)
        batch_tensor=torch.from_numpy(pad_batch).long()
    return batch_tensor

def test_collate_fn(batch, sep_id):
    # prediction
    # sep_id: the token id that divides input context and target context
    inputs, labels = [], []
    for seq in batch:
        idx=seq.index(sep_id)
        inputs.append(seq[:idx+1])
        labels.append(seq[idx+1:])
    return [inputs, labels]

def padSeqs(sequences, pad_id, maxlen=None):
    lengths = [len(x) for x in sequences]
    maxlen=max(lengths)
    maxlen=min(512, maxlen)
    
    pad_batch=np.ones((len(sequences), maxlen))*pad_id
    for idx, s in enumerate(sequences):
        trunc = s[-maxlen:]
        pad_batch[idx, :len(trunc)] = trunc
            
    return pad_batch

def integrate_result(inputs, gens, oracles):
    results=[]
    for context, gen, oracle in zip(inputs, gens, oracles):
        EN_list, left=context.split('[EOS_L]')
        EN_list=EN_list.strip('[CLS]')
        user, left = left.split('[EOS_U]')
        EN, left=left.split('[EOS_E]')
        user_intent, KB_result=left.split('[EOS_UI]')
        KB_result=KB_result.strip('[EOS_K]')
        service_intent, service=oracle.split('[EOS_SI]')
        service=service[:service.index('[EOS_S]')]
        if '[EOS_SI]' in gen:
            service_intent_gen, service_gen=gen.split('[EOS_SI]')
            service_gen=service_gen[:service_gen.index('[EOS_S]')]
        else:
            service_intent_gen=''
            service_gen=gen[:gen.index('[EOS_S]')]
        entry={
            '用户':user.replace(' ', ''),
            '用户意图':user_intent.replace(' ', ''),
            '实体列表':EN_list.replace(' ', ''),
            '实体':EN.replace(' ', ''),
            '数据库结果':KB_result.replace(' ', ''),
            '客服':service.replace(' ', ''),
            '客服意图':service_intent.replace(' ', ''),
            '客服-生成':service_gen.replace(' ', ''),
            '客服意图-生成':service_intent_gen.replace(' ', '')
        }

        results.append(entry)
    return results

def batch_align(contexts,left_len=0,return_attn=False): # left padding
    max_len=max([len(context) for context in contexts])
    max_len=min(512, max_len) # 1024-left_len
    new_contexts=[]
    attentions=[]
    for id, context in enumerate(contexts):
        if len(context)<max_len:
            new_context=(max_len-len(context))*[cfg.pad_id]+context
            attention=(max_len-len(context))*[0]+len(context)*[1]
        else:
            new_context=context[-max_len:]
            attention=len(new_context)*[1]
        new_contexts.append(new_context)
        attentions.append(attention)
    if return_attn:
        return new_contexts, attentions
    return new_contexts

if __name__=='__main__':
    #data = json.load(open(cfg.data_path,'r'))
    #get_retrieval_data(data)

    train_data=json.load(open(cfg.data_path, 'r', encoding='utf-8'))
    train_ids = [dial['id'] for dial in train_data]
    train_seqs=convert_to_sequences(train_data, train_ids, return_dict=True)
    """
    data=json.load(open(cfg.data_path, 'r', encoding='utf-8'))
    dial_ids=list(data.keys())
    random.shuffle(dial_ids)
    piece=len(dial_ids)//10
    train_ids, dev_ids, test_ids=dial_ids[:8*piece], dial_ids[8*piece:9*piece], dial_ids[9*piece:]
    train_seqs=convert_to_sequences(data, train_ids)
    dev_seqs=convert_to_sequences(data, dev_ids)
    test_seqs=convert_to_sequences(data, test_ids)
    print('Train:{}, dev:{}, test:{}'.format(len(train_seqs), len(dev_seqs), len(test_seqs)))
    seq_data={
        'train':train_seqs,
        'dev':dev_seqs,
        'test':test_seqs
    }
    json.dump(seq_data, open(os.path.join(cfg.data_dir, 'sequences.json'), 'w'), indent=2, ensure_ascii=False)
    """