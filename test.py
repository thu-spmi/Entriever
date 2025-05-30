
from transformers import GPT2LMHeadModel
from transformers import BertTokenizer
from reader import *
from metrics import *
from main import Model
from reader import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os, shutil
import random
import argparse
import time
import logging
import json
from tqdm import tqdm
import numpy as np
import copy, re
from torch.utils.tensorboard import SummaryWriter

#not used, can be used if main and model are designed differently

class Semi_supervision(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device=cfg.device
        self.tokenizer = BertTokenizer.from_pretrained(cfg.gpt_path)

        if isinstance(self.device,list):
            self.PrioriModel=Model(self.device[0])#GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
            self.PosteriorModel=Model(self.device[1],posterior=True)

        json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
    
        #init_vocab_size=len(self.tokenizer)
        #special_tokens_dict = {'additional_special_tokens': special_tokens}
        #logging.info('Added special tokens:{}'.format(special_tokens))
        #self.tokenizer.add_special_tokens(special_tokens_dict)
        #self.model.resize_token_embeddings(len(self.tokenizer))
        logging.info('Special token added, vocab size:{}'.format( len(self.tokenizer)))

        # log
        log_path='./log/log_{}'.format(cfg.exp_name)
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
            os.mkdir(log_path)
        else:
            os.mkdir(log_path)
        self.tb_writer = SummaryWriter(log_dir=log_path)

    def jsa_train(self):
        cfg = self.cfg
        logging.info('------Running joint stochastic approximation------')
        ratio = 1 # hyperparameter for num of jsa-training and supervised-training, type int

        encoded_data=read_data(self.tokenizer)
        #encoded_data_post=read_data(self.tokenizer,posterior=True)
        #train_dataloader=DataLoader(encoded_data['train'], batch_size=cfg.batch_size, shuffle=True, collate_fn=train_collate_fn) 
        #train_post_dataloader=DataLoader(encoded_data_post['train'], batch_size=cfg.batch_size, shuffle=True, collate_fn=train_collate_fn)
        #dev_dataloader=DataLoader(encoded_data['dev'], batch_size=cfg.eval_batch_size, collate_fn=train_collate_fn)
        
        sup_num = len(encoded_data['train'])
        num_dials=(1+ratio)*sup_num

        steps_per_epoch = sup_num//cfg.batch_size
        logging.info('Labeled dials:{}, ratio:{}, steps:{}'.format(sup_num,ratio))
        
        cfg.batch_size=cfg.batch_size*cfg.gradient_accumulation_steps

        batches_unl=get_unsup()
        #unlabel_turns=set_stats['train']['num_turns']
        all_batches=[]
        all_batches_repeat=[]

        if cfg.debugging:
            batches_unl=batches_unl[:len(batches_unl)//15]

        for batch in batches_unl:
            all_batches.append({'batch':self.reader.transpose_batch(batch),'supervised':False})
            all_batches_repeat.append({'batch':self.reader.transpose_batch(batch),'supervised':False})
        batch_num=sum([len(item['batch']) for item in all_batches])

        optimizer1, scheduler1 = self.PrioriModel.get_optimizers(num_dials, self.PrioriModel)
        optimizer2, scheduler2 = self.PosteriorModel.get_optimizers(num_dials, self.PosteriorModel)

        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info('  Num Batches = %d', len(all_batches))
        log_inputs = 3
        global_step = 0
        max_score=0
        
        weight_decay_count=cfg.weight_decay_count
        lr=cfg.lr
        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss, sup_loss, uns_loss = 0.0, 0.0, 0.0
            sup_step, uns_step=0, 0
            btm = time.time()
            self.PrioriModel.model.zero_grad()
            self.PosteriorModel.model.zero_grad()

            random.shuffle(all_batches_repeat)

            for batch_idx in range(steps_per_epoch):

                #unsup_train
                for _ in range(ratio):
                    for turn_num, turn_batch in enumerate(batches_unl):
                        turn_batch, next_pv_batch=self.gen_hidden_state(turn_batch, pv_batch, turn_num, posterior=True)
                    
                    first_turn = (turn_num == 0)
                    mini_batches, mini_pv_batches=self.reader.split_turn_batch(turn_batch, cfg.origin_batch_size, other_batch=pv_batch)
                    for i, batch in enumerate(mini_batches):
                        mini_pv_batch=None if turn_num==0 else mini_pv_batches[i]
                        inputs_prior, labels_prior, seg_labels = convert_batch_turn(batch, mini_pv_batch, first_turn, posterior=False, seg_label=True)
                        inputs_posterior, labels_posterior = convert_batch_turn(batch, mini_pv_batch, first_turn, posterior=True)
                        self.PrioriModel.train()
                        self.PosteriorModel.train()
                        if log_inputs > 0 and cfg.example_log:  # log inputs for the very first two turns
                            logging.info('Prior examples:\n{}'.format(self.tokenizer.decode(inputs_prior['contexts'][0])))
                            logging.info("Posterior examples:\n{}".format(self.tokenizer.decode(inputs_posterior['contexts'][0])))
                            log_inputs -= 1

                        jsa_labels=(copy.deepcopy(inputs_posterior),copy.deepcopy(labels_posterior),copy.deepcopy(inputs_prior),copy.deepcopy(labels_prior))
                        # to tensor
                        inputs_prior = self.add_torch_input(inputs_prior)#B,T
                        inputs_posterior = self.add_torch_input(inputs_posterior,posterior=True)
                        labels_prior=self.add_torch_input(labels_prior)#B,T
                        labels_posterior=self.add_torch_input(labels_posterior,posterior=True)
                        # loss
                        outputs_prior=self.PrioriModel(inputs_prior['contexts_tensor'])
                        outputs_posterior=self.PosteriorModel(inputs_posterior['contexts_tensor'])#B,T,V
                        logits_pri=outputs_prior[0]
                        logits_post=outputs_posterior[0]
                        
                        #get prob
                        jsa_prob=self.get_jsa_prob(logits_pri,logits_post,\
                                labels_prior['contexts_tensor'],labels_posterior['contexts_tensor'])
                        if epoch==0:
                        #if epoch>-1:
                            last_prob=jsa_prob #accept the proposal at the first turn
                            #if 'prob' not in turn_batch:
                            #    turn_batch['prob']=[]
                            #turn_batch['prob'].append(jsa_prob)
                        else:
                            
                            t_label=turn_batch['jsa_labels'][i]
                            temp_label=copy.deepcopy(t_label)
                            i_posterior=self.add_torch_input(temp_label[0],posterior=True)
                            l_posterior=self.add_torch_input(temp_label[1],posterior=True)
                            i_prior=self.add_torch_input(temp_label[2])
                            l_prior=self.add_torch_input(temp_label[3])
                            o_prior=self.PrioriModel(i_prior['contexts_tensor'])
                            o_posterior=self.PosteriorModel(i_posterior['contexts_tensor'])#B,T,V
                            lo_pri=o_prior[0]
                            lo_post=o_posterior[0]
                            
                            #get prob
                            last_prob=self.get_jsa_prob(lo_pri,lo_post,\
                                    l_prior['contexts_tensor'],l_posterior['contexts_tensor'])
                            
                            #last_prob=copy.deepcopy(turn_batch['prob'][i])
                            #turn_batch['prob'][i]=jsa_prob
                            #last_prob=jsa_prob
                        
                        #update bspn
                        for prob_num in range(min(len(jsa_prob),len(last_prob))):
                            if jsa_prob[prob_num]-last_prob[prob_num]>0:
                                ratio=1.0
                            else:
                                ratio=math.exp(jsa_prob[prob_num]-last_prob[prob_num])
                            if ratio<1.0:
                                if random.random()>ratio:
                                    for j in range(4):
                                        if 'contexts_np' in jsa_labels[j]:
                                            jsa_labels[j].pop('contexts_np')
                                        jsa_labels[j]['contexts'][prob_num]=turn_batch['jsa_labels'][i][j]['contexts'][prob_num]
                                        #jsa_labels[j]['contexts_np'][prob_num]=dial_batch_dict['jsa_labels'][j]['contexts_np'][prob_num]
                                        jsa_labels[j]['lengths'][prob_num]=turn_batch['jsa_labels'][i][j]['lengths'][prob_num]                        
                        if epoch==0:
                            if 'jsa_labels' not in turn_batch:
                                turn_batch['jsa_labels']=[]
                            turn_batch['jsa_labels'].append(jsa_labels)
                        else:
                            turn_batch['jsa_labels'][i]=jsa_labels
                        temp_label=copy.deepcopy(jsa_labels)
                        inputs_posterior=self.add_torch_input(temp_label[0],posterior=True)
                        labels_posterior=self.add_torch_input(temp_label[1],posterior=True)
                        inputs_prior=self.add_torch_input(temp_label[2])
                        labels_prior=self.add_torch_input(temp_label[3])
                        if epoch==0:
                        #if epoch>-1:
                            #straight through trick
                            #ST_inputs_prior, resp_label=self.get_ST_input1(inputs_prior['contexts_tensor'], logits_post, labels_prior['contexts_tensor'], list(seg_labels['contexts_np']))
                            ST_inputs_prior, resp_label=self.get_ST_input(inputs_prior['contexts_tensor'],\
                                logits_post,labels_prior['contexts_tensor'],labels_posterior['contexts_tensor'])
                            embed_prior=ST_inputs_prior.matmul(self.PrioriModel.get_input_embeddings().weight)#multiple the input embedding
                            outputs1=self.PrioriModel(inputs_embeds=embed_prior)    
                            #outputs1=self.PrioriModel(inputs_prior['contexts_tensor'])
                            loss_pri=self.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                            #loss_pri=self.calculate_loss_and_accuracy(outputs1,resp_label)
                        else:
                            outputs1=self.PrioriModel(inputs_prior['contexts_tensor'])
                            loss_pri=self.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                            outputs2=self.PosteriorModel(inputs_posterior['contexts_tensor'])
                            loss_pos=self.calculate_loss_and_accuracy(outputs2,labels_posterior['contexts_tensor'])
                        
                        if cfg.loss_reg:
                            loss_pri=loss_pri/cfg.gradient_accumulation_steps
                            loss_pos=loss_pos/cfg.gradient_accumulation_steps
                        st3=0
                        loss_pri.backward()
                        #if epoch<0:
                        if epoch!=0:
                            loss_pos.backward()
                            loss=loss_pri+loss_pos.to(self.device1)
                            tr_loss += loss.item()
                            uns_loss += loss.item()
                        else :
                            loss=loss_pri
                            tr_loss += loss_pri.item()
                            uns_loss += loss_pri.item()
                        uns_step+=1
                        torch.nn.utils.clip_grad_norm_(self.PrioriModel.parameters(), 5.0)
                        torch.nn.utils.clip_grad_norm_(self.PosteriorModel.parameters(), 5.0)
                    epoch_step+=1
                    optimizer1.step()
                    optimizer1.zero_grad()
                    optimizer2.step()
                    optimizer2.zero_grad()
                    global_step+=1
                    if cfg.use_scheduler:
                        scheduler1.step()
                        scheduler2.step()
                    if self.tb_writer:
                        self.tb_writer.add_scalar('lr1', optimizer1.param_groups[0]["lr"],global_step)
                        self.tb_writer.add_scalar('lr2', optimizer2.param_groups[0]["lr"],global_step)
                        self.tb_writer.add_scalar('loss', loss.item(), global_step)
                    pv_batch=next_pv_batch
            
            logging.info('Epoch: {}, Train epoch time: {:.2f} min, loss:{:.3f}, avg_sup_loss:{:.3f}, avg_uns_loss:{:.3f}'.format(epoch, 
                (time.time()-btm)/60, tr_loss/epoch_step, sup_loss/(sup_step+1e-10), uns_loss/(uns_step+1e-10)))
            eval_result=self.validate_fast(data='dev')
            if self.tb_writer:
                self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                self.tb_writer.add_scalar('match',eval_result['match'],epoch)
                self.tb_writer.add_scalar('success',eval_result['success'],epoch)
                self.tb_writer.add_scalar('bleu',eval_result['bleu'],epoch)
                self.tb_writer.add_scalar('combined_score',eval_result['score'],epoch)
            
            if eval_result['score']>max_score:
                max_score=eval_result['score']
                self.save_model(path='best_score_model')
                self.save_model(path='best_post_model',posterior=True)
            if epoch==0 and cfg.save_first_epoch:
                self.save_model(path='first_epoch_model')
            else:
                weight_decay_count-=1
                if weight_decay_count==0 and not cfg.use_scheduler:
                    lr=lr*cfg.lr_decay
                    for group in optimizer1.param_groups:
                        group['lr'] = lr
                    for group in optimizer2.param_groups:
                        group['lr'] = lr
                    logging.info("learning rate decay to {}".format(lr))
                    weight_decay_count = cfg.weight_decay_count
            if lr<1e-9 and not cfg.use_scheduler:
                logging.info('learning rate too small, break')
                break    
def test():
    #data=json.load(open('data/seretod/test_data.json', 'r', encoding='utf-8'))
    data=json.load(open('data/processed_data.json', 'r', encoding='utf-8'))
    special_turns = []
    notify_turns = []
    confirm = []
    confirm_num = 0 # 14520
    notify_num = 0 #52974
    count = 0 # 28307
    kb_num = 0 # 4129
    from metrics import extract_request_info
    for dial in data:
        request_info = extract_request_info(dial['goal'],dial['KB'])
        if request_info!=[]:
            kb_num = kb_num + 1
        r_info = []
        for i in request_info:
            r_info.extend(i.split(','))
        for turn in dial['content']:
            if '通知' in turn['客服意图']:
                notify_num = notify_num + 1
                #for triple in turn['info']['triples']:
                for info in r_info:
                    if (info in turn['客服']) and (turn not in notify_turns):
                        notify_turns.append(turn)
                        break
            if ('询问' not in turn['用户意图'])  and ('求助-查询' not in turn['用户意图']):
                if '主动确认' not in turn['用户意图']:
                    #for triple in turn['info']['triples']:
                    for info in r_info:
                        if (info in turn['客服']) and (turn not in special_turns):
                            special_turns.append(turn)
                            if '通知' in turn['客服意图']:
                                count = count + 1
                            break
                else:
                    confirm_num = confirm_num + 1
                    #for triple in turn['info']['triples']:
                    for info in r_info:
                        if (info in turn['客服']) and (turn not in confirm) and (info not in turn['用户'])and ('通知' not in turn['客服意图']):
                            confirm.append(turn)
                            break
            else: 
                if('(') not in turn['用户意图']:
                    for info in r_info:
                        if (info in turn['客服']) and (turn not in special_turns):
                            special_turns.append(turn)
                            if '通知' in turn['客服意图']:
                                count = count + 1
                            break

                


    #data=json.load(open('data/seretod/test_data_for_track2.json', 'r', encoding='utf-8'))
    #from metrics import extract_request_info
    #for dial in data:
    #    required = extract_request_info(dial['goal'],dial['KB'])
    #    dial['required'] = required
    #json.dump(data,open('data/seretod/test_data.json', 'w'),indent=2, ensure_ascii=False)
    print(1)
if __name__ == "__main__":
    #test()
    data=json.load(open('data/seretod/dst_data.json', 'r', encoding='utf-8'))
    #data = json.load(open('data/all_data.json', 'r', encoding='utf-8'))
    #data=json.load(open(cfg.data_path, 'r', encoding='utf-8')) # cfg.data_path
    count = 0
    dialogs =[]
    special_turn = []
    wanted_turn = []
    copy_turn = []
    intents = {}
    all_sys_intents={}
    all_info = []
    all_intent = ['问候', '提供信息', '被动确认', '客套', '主动确认', '求助-查询', '其他', '投诉反馈', '否认', '求助-故障', '询问', '取消', '请求重复', '再见']
    all_sys_intent = ['通知', '引导', '其他', '再见', '被动确认', '询问', '客套', '主动确认', '建议', '抱歉', '请求重复','问候', '否认']
    for dial in data:
        dial_id=dial['id']   
        kb = dial['KB']
        goal = extract_request_info(dial['goal'],dial['KB'])
        for turn in dial['content']:
            ui = turn['用户意图']
            si = turn['客服意图']
            user = turn['用户']
            sys = turn['客服']
            """
            for ent,props in kb.items():
                for key,value in props.items():
                    if (value in sys) and(turn not in wanted_turn):
                        if value in user and turn not in copy_turn:
                            copy_turn.append(turn)
                        else:
                            wanted_turn.append(turn)
            for sys_intent in si.split(','):
                if sys_intent not in all_sys_intent:
                    if sys_intent not in all_sys_intents:
                        all_sys_intents[sys_intent] = 0
                    all_sys_intents[sys_intent] =  all_sys_intents[sys_intent] + 1
            """
            for intent in ui.split(','):
                if '(' in intent:
                    infos=re.findall(r'\((.*?)\)', intent)
                    count = count + 1
                    if len(infos)>1:
                        special_turn.append(turn)
                    for info in infos:
                        if 'ent' in info:
                            info = info[6:]
                        if info not in all_info:
                            all_info.append(info)
            if si == '被动确认':
                count =count + 1
                for intent in ui.split(','):
                    if ('询问' in intent or '求助-查询' in intent) and (turn not in wanted_turn):
                        wanted_turn.append(turn)
    """
    user_acts = {}
    sys_acts = {}
    dialogs.extend(data['test'])
    dialogs.extend(data['train'])
    dialogs.extend(data['dev'])
    for dialog in dialogs:
        for turn in dialog['log']:
            bspn = turn['bspn'].replace('[EOS_UI]','').split(',')
            for intent in bspn:
                if intent not in user_acts:
                    user_acts[intent] = 0
                user_acts[intent] = user_acts[intent] + 1
            aspn = turn['aspn'].replace('[EOS_SI]','').split(',')
            for act in aspn:
                if act not in sys_acts:
                    sys_acts[act]= 0
                sys_acts[act] = sys_acts[act] + 1
    """
    json.dump(data, open('data/postprocessed_data.json', 'w'),indent=2, ensure_ascii=False)
    print(1)
    
    #cfg = None
    #semi = Semi_supervision(cfg)
