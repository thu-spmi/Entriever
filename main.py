
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2LMHeadModel, BertModel, BertForNextSentencePrediction, MT5ForConditionalGeneration
from transformers import BertTokenizer, T5Tokenizer
from reader import *
from metrics import *
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
from config import global_config as cfg
from retrieve_kb import EncoderModel, EBM, collate_fn, get_optimizers

class Model(object):
    def __init__(self, device=[0],posterior = False):
        self.posterior = posterior
        if isinstance(device,list):
            self.device = device[0]
        else:
            self.device = device
        if posterior:
            self.model=GPT2LMHeadModel.from_pretrained(cfg.posterior_path) if cfg.gpt else MT5ForConditionalGeneration.from_pretrained(cfg.t5_posterior_path)
            self.tokenizer = BertTokenizer.from_pretrained(cfg.posterior_path) if cfg.gpt else BertTokenizer.from_pretrained(cfg.t5_posterior_path)
        else:
            self.model=GPT2LMHeadModel.from_pretrained(cfg.gpt_path) if cfg.gpt else MT5ForConditionalGeneration.from_pretrained(cfg.t5_path)
            self.tokenizer = BertTokenizer.from_pretrained(cfg.gpt_path) if cfg.gpt else BertTokenizer.from_pretrained(cfg.t5_path)
        self.model.to(self.device)

        if 'train' in cfg.mode:
            json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
        
            # Add special tokens
            init_vocab_size=len(self.tokenizer)
            special_tokens_dict = {'additional_special_tokens': special_tokens}
            logging.info('Added special tokens:{}'.format(special_tokens))
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
            logging.info('Special token added, vocab size:{}-->{}'.format(init_vocab_size, len(self.tokenizer)))

        # log
        log_path='./log/log_{}'.format(cfg.exp_name)
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
            os.mkdir(log_path)
        else:
            os.mkdir(log_path)
        self.tb_writer = SummaryWriter(log_dir=log_path)
    
    def train(self):
        if not cfg.only_target_loss:
            if (cfg.mode == 'train' or cfg.mode == 'train_post'):
                encoded_data=read_data(self.tokenizer,self.posterior)
            if cfg.mode == 'pretrain':
                encoded_data=get_unsup(self.tokenizer,pretrain=True)
            train_dataloader=DataLoader(encoded_data['train'], batch_size=cfg.batch_size, shuffle=True, collate_fn=train_collate_fn) 
            dev_dataloader=DataLoader(encoded_data['dev'], batch_size=cfg.eval_batch_size, collate_fn=train_collate_fn)
            optimizer, scheduler = self.get_optimizers(len(encoded_data['train']), self.model)
        else:
            encoded_data=read_data(self.tokenizer,return_dict=True)
            train_data=encoded_data['train']
            random.shuffle(train_data)
            train_dataloader = get_batches(train_data,cfg.batch_size)
            #num_dials=len(train_data)
            num_turns = sum(len(dial) for dial in train_data)
            optimizer, scheduler = self.get_optimizers(num_turns, self.model)
            cls = self.tokenizer.convert_tokens_to_ids('[CLS]')
            sep = self.tokenizer.convert_tokens_to_ids('[SEP]')

        if cfg.debugging:
            train_dataloader = train_dataloader[:len(train_dataloader)//40]
        log_inputs = 2
        global_step = 0
        min_loss=10000
        max_score=0

        if cfg.joint_training:
            bert_tokenizer = BertTokenizer.from_pretrained(cfg.bert_save_path)
            if cfg.only_one_model:
                bert_model = BertForNextSentencePrediction.from_pretrained(cfg.bert_save_path)#EncoderModel(cfg,tokenizer)
                bert_model.to(cfg.device[0])
            else:
                bert_model = EncoderModel(cfg,bert_tokenizer)
            if cfg.fix_retrieval_model:
                bert_model.eval()
        for epoch in range(cfg.epoch_num):
            tr_loss = 0.0
            step_loss = 0
            epoch_step = 0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()
            for batch_idx, batch in enumerate(train_dataloader):
                self.model.train()
                try:  # avoid OOM
                    if cfg.only_target_loss: # default setting
                        dial_batch=transpose_batch(batch)
                        pv_batch = None
                        spoken = []
                        for turn_num, turn_batch in enumerate(dial_batch):
                            first_turn = (turn_num == 0)
                            if cfg.gpt:
                                # cfg.fix_retrieval_model
                                if cfg.joint_training and ('db_retrieval' not in turn_batch):
                                    user = self.decode_batch(turn_batch['user'])
                                    sys = self.decode_batch(turn_batch['resp'])
                                    spoken, hist = self.get_spoken(spoken, user, role='user')
                                    turn_batch['db_retrieval'], _ = self.query_KB_retrieval(bert_model, bert_tokenizer, spoken, hist, turn_batch['KB'])
                                    spoken, _=self.get_spoken(spoken, sys, role='system')
                                inputs, labels = convert_batch_turn(cls,sep,turn_batch, pv_batch, first_turn, posterior=self.posterior) 
                                if cfg.mix_retrieval_training: # must be used in joint training mode
                                    cfg.joint_training=False
                                    inputs1, labels1 = convert_batch_turn(cls,sep,turn_batch, pv_batch, first_turn, posterior=self.posterior) 
                                    inputs1 = self.add_torch_input(inputs1)
                                    labels1 = self.add_torch_input(labels1) 
                                    outputs = self.model(inputs1['contexts_tensor'])   
                                    loss = self.calculate_loss_and_accuracy(outputs, labels=labels1['contexts_tensor'])
                                    cfg.joint_training=True
                                # previously, pv_batch is not used in training, using ground truth
                                # now, pv_batch means last_turn
                                pv_batch = self.get_pv_batch(pv_batch, turn_batch['user'], turn_batch['resp'],  turn_batch['entity'])
                                if log_inputs > 0:  # log inputs for the very first two turns
                                    logging.info('Input examples:\n{}'.format(self.tokenizer.decode(inputs['contexts'][0])))
                                    log_inputs-=1
                                inputs = self.add_torch_input(inputs)
                                labels = self.add_torch_input(labels) 
                                outputs = self.model(inputs['contexts_tensor'])
                                if cfg.mix_retrieval_training:  
                                    loss = loss + self.calculate_loss_and_accuracy(outputs, labels=labels['contexts_tensor']) 
                                else:
                                    loss = self.calculate_loss_and_accuracy(outputs, labels=labels['contexts_tensor'])
                            else:
                                inputs, labels = convert_batch_t5(cls,sep,turn_batch, turn_batch['ent_list'], first_turn, posterior=self.posterior)
                                pv_batch = self.get_pv_batch(pv_batch, turn_batch['user'], turn_batch['resp'],  turn_batch['entity'])
                                if log_inputs > 0:  # log inputs for the very first two turns
                                    logging.info('Input examples:\n{}'.format(self.tokenizer.decode(inputs['contexts'][0])))
                                    logging.info('Output examples:\n{}'.format(self.tokenizer.decode(labels['contexts'][0])))
                                    log_inputs-=1
                                inputs = self.add_torch_input(inputs)
                                labels = self.add_torch_input(labels)
                                labels['contexts_tensor'][labels['contexts_tensor'] == self.tokenizer.pad_token_id] = -100
                                loss = self.model(input_ids=inputs['contexts_tensor'], attention_mask=inputs['attention_tensor'], labels=labels['contexts_tensor'], return_dict=False)[0]
                            loss.backward()
                            tr_loss += loss.item()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                            epoch_step += 1

                            # step, wrt gradient_accumulation_steps, clip grad norm
                            if epoch_step % cfg.gradient_accumulation_steps == 0 or(
                                batch_idx==len(train_dataloader)-1 and turn_num==len(dial_batch)-1):
                                optimizer.step()
                                scheduler.step()
                                optimizer.zero_grad()
                                # global_step: actual step the optimizer took
                                global_step += 1
                    else:
                        if log_inputs > 0:  # log inputs for the very first two turns
                            logging.info('Training Sequences:')
                            if cfg.gpt:
                                logging.info(self.tokenizer.decode(batch[0,:]))
                            else:
                                logging.info(self.tokenizer.decode(batch['input'][0,:]))
                            log_inputs-=1
                        if cfg.gpt:
                            inputs=batch.to(self.device) #B, T
                            labels=inputs
                            outputs = self.model(inputs)
                            loss = self.calculate_loss_and_accuracy(outputs, labels=labels)
                        else:
                            for key, val in batch.items():
                                if type(batch[key]) is list:
                                    continue
                                batch[key] = batch[key].to(self.device)
                            batch['output'][batch['output'] == self.tokenizer.pad_token_id] = -100
                            loss = self.model(input_ids=batch['input'], labels=batch['output'], return_dict=False)[0]
                        loss=loss/cfg.gradient_accumulation_steps
                        loss.backward()
                        tr_loss += loss.item()
                        step_loss+=loss.item()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                        if (batch_idx+1) % cfg.gradient_accumulation_steps == 0 or batch_idx+1==len(train_dataloader):
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            global_step += 1
                            self.tb_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], global_step)
                            self.tb_writer.add_scalar('loss', step_loss, global_step)
                            step_loss=0

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        oom_time += 1
                        logging.info("WARNING: ran out of memory,times: {}, batch size: {}".format(oom_time, cfg.batch_size))
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        logging.info(str(exception))
                        raise exception
            if not cfg.only_target_loss:
                eval_loss=self.eval(dev_dataloader)
            if cfg.save_type =='max_score':
                if self.posterior:
                    eval_result=self.validate_post()
                    ui = eval_result['P/R/F1 for user intent'][2]
                    si = eval_result['P/R/F1 for system intent'][2]
                    db = eval_result['P/R/F1 for db prediction'][2] 
                    eval_loss = ui + si + db
                    logging.info('user:{:.3f}, system:{:.3f} , db:{:.3f}, score:{:.3f}'.format(ui, si, db, eval_loss))
                else:
                    cfg.gt_db = False
                    cfg.retrieve_kb = True
                    eval_result=self.validate_fast() # 'test'
                    cfg.retrieve_kb = False
                    cfg.gt_db = True
                    ui = eval_result['P/R/F1 for user intent'][2]
                    si = eval_result['P/R/F1 for system intent'][2]
                    bleu = eval_result['BLEU']
                    success = eval_result['Success']
                    logging.info('user:{:.3f}, system:{:.3f} , bleu:{:.3f}, success:{:.3f}'.format(ui, si, bleu, success))
                    eval_loss = bleu/50 + success
            logging.info('Epoch:{}, Train epoch time:{:.2f} min, epoch loss:{:.3f}, eval loss:{:.3f}'.format(epoch, (time.time()-btm)/60, tr_loss, eval_loss))
            self.tb_writer.add_scalar('eval_loss', eval_loss, epoch)
            if cfg.save_type =='max_score':
                if max_score < eval_loss:
                    max_score=eval_loss
                    self.save_model()
            else:
                if eval_loss<min_loss:
                    min_loss=eval_loss
                    self.save_model()
        #self.save_model(path='last_epoch_model')

    def train_one_step(self,batches,optimizer,scheduler):
        tr_loss = 0.0
        step_loss=0
        oom_time = 0
        for batch_idx, batch in enumerate(batches):
            try:  # avoid OOM
                self.model.train()
                #if log_inputs > 0:  # log inputs for the very first two turns
                #    logging.info('Training Sequences:')
                #    logging.info(self.tokenizer.decode(batch[0,:]))
                #    log_inputs-=1
                inputs=batch.to(self.device) #B, T
                labels=inputs
                outputs = self.model(inputs)
                loss = self.calculate_loss_and_accuracy(outputs, labels=labels)
                loss=loss/cfg.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                step_loss+=loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                if (batch_idx+1) % cfg.gradient_accumulation_steps == 0 or batch_idx+1==len(batches):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    #self.tb_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], global_step)
                    #self.tb_writer.add_scalar('loss', step_loss, global_step)
                    step_loss=0
                    #torch.cuda.empty_cache()
                    #schedule_count = 0
                #else:
                    #schedule_count = schedule_count + 1

            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    oom_time += 1
                    logging.info("WARNING: ran out of memory,times: {}, batch size: {}".format(oom_time, cfg.batch_size))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    logging.info(str(exception))
                    raise exception
        return tr_loss

    def get_optimizers(self, num_samples, model):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        #print(num_samples, cfg.epoch_num, cfg.gradient_accumulation_steps, cfg.batch_size)
        num_training_steps = num_samples*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.origin_batch_size) # origin_batch_size_here
        num_warmup_steps = int(num_training_steps*cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,\
            num_training_steps=num_training_steps)
        return optimizer, scheduler
    
    def calculate_loss_and_accuracy(self, outputs, labels):
        # GPT2-chicahat/train.py
        lm_logits = outputs[0]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=cfg.pad_id, reduction='sum')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # avg loss
        not_ignore = shift_labels.ne(cfg.pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
        return loss
    
    def eval(self, data):
        self.model.eval()
        total_loss=0
        with torch.no_grad():
            for batch in data:
                if cfg.gpt:
                    inputs=batch.to(self.device) #B, T
                    labels=inputs
                    outputs = self.model(inputs)
                    loss = self.calculate_loss_and_accuracy(outputs, labels=labels)
                else:
                    for key, val in batch.items():
                        if type(batch[key]) is list:
                            continue
                        batch[key] = batch[key].to(self.device)
                    batch['output'][batch['output'] == self.tokenizer.pad_token_id] = -100
                    loss = self.model(input_ids=batch['input'], labels=batch['output'], return_dict=False)[0]
                total_loss+=loss.item()
        return total_loss/len(data)

    def save_model(self, path=None, model=None):
        if self.posterior:
            save_path = os.path.join(cfg.exp_path, path) if path else os.path.join(cfg.exp_path, 'best_post_model')
        else:
            save_path = os.path.join(cfg.exp_path, path) if path else os.path.join(cfg.exp_path, 'best_model')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info('Saving model checkpoint to %s', save_path)
        if not model:
            self.model.save_pretrained(save_path)
        else:
            model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def add_torch_input(self, inputs):
        # to tensor and to device
        if 'contexts_np' not in inputs:
            inputs['contexts_np'],_=padSeqs_gpt(inputs['contexts'], cfg.pad_id)
        # add attention if attention in inputs
        if 'attention' in inputs:
            attentions_tensor = torch.from_numpy(inputs['attention']).long()
            attentions_tensor = attentions_tensor.to(self.device)
            inputs['attention_tensor'] = attentions_tensor

        contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
        contexts_tensor = contexts_tensor.to(self.device)
        inputs['contexts_tensor'] = contexts_tensor
        return inputs

    def test_end_to_end(self, data='test'):
        self.model.eval()
        result_path=os.path.join(cfg.gpt_path, 'e2e_result.json')
        if os.path.exists(result_path):
            test_data=json.load(open(result_path, 'r', encoding='utf-8'))
        else:#generate results
            test_data=extract_test_dial(data=data)
            st=time.time()
            dial_num=0
            turn_num=0
            with torch.no_grad():
                for dial in tqdm(test_data):
                    dial_num+=1
                    #if dial_num==5:
                    #   break
                    KB, goal=dial['KB'], dial['goal']
                    EN_list=set([])
                    for turn in dial['content']:
                        turn_num+=1
                        pv_EN_list=copy.deepcopy(EN_list)
                        EN_list_seq=','.join(list(pv_EN_list))
                        context=EN_list_seq+'[EOS_L]'+turn['用户'].lower()+'[EOS_U]'
                        context_ids=self.tokenizer.encode(context)[:-1]
                        # predict entity names mentioned in this turn
                        #logging.info(self.tokenizer.decode(context_ids).replace(' ', ''))
                        max_len=len(context_ids)+15
                        eos_id=self.tokenizer.convert_tokens_to_ids('[EOS_E]')
                        outputs = self.model.generate(input_ids=torch.tensor([context_ids]).to(self.model.device),
                                                pad_token_id=cfg.pad_id, max_length=max_len, eos_token_id=eos_id)
                        generated = outputs[0].cpu().numpy().tolist()
                        if eos_id not in generated:
                            generated[-1]=eos_id
                        EN=self.tokenizer.decode(generated[len(context_ids):-1]).replace(' ', '')
                        # delete repetition
                        current_EN_set=set(EN.split(','))
                        EN=','.join(list(current_EN_set))
                        if EN!='':
                            EN_list=EN_list.union(current_EN_set)
                        # predict user intent
                        context_ids=generated
                        #logging.info(self.tokenizer.decode(context_ids).replace(' ', ''))
                        max_len=len(context_ids)+25
                        eos_id=self.tokenizer.convert_tokens_to_ids('[EOS_UI]')
                        outputs = self.model.generate(input_ids=torch.tensor([context_ids]).to(self.model.device),
                                                pad_token_id=cfg.pad_id, max_length=max_len, eos_token_id=eos_id)
                        generated = outputs[0].cpu().numpy().tolist()
                        if eos_id not in generated:
                            generated[-1]=eos_id
                        UI=self.tokenizer.decode(generated[len(context_ids):-1]).replace(' ', '')
                        # query local database
                        KB_result=[]
                        if '(' in UI:
                            for intent in UI.split(','):
                                if '('  not in intent:
                                    continue
                                act=intent[:intent.index('(')]
                                info=re.findall(r'\((.*?)\)', intent)
                                for e in info:
                                    e=e.strip('-')
                                    if '-' in e:
                                        if len(e.split('-'))!=2:
                                            continue
                                        ent_name, prop=e.split('-')
                                        res=query(KB, ent_name=ent_name, prop=prop)
                                    elif e.lower() in ['业务','数据业务','套餐', '主套餐','附加套餐','国际漫游业务','流量包','长途业务','4g套餐','5g套餐']:
                                        res=query(KB, ent_type=e)
                                    else:
                                        res=query(KB, prop=e)
                                    if res is not None:
                                        if isinstance(res, list):
                                            KB_result.append(','.join(res))
                                        else:
                                            KB_result.append(res)
                        KB_seq=','.join(KB_result)
                        # generate system intent
                        context_ids=generated+self.tokenizer.encode(KB_seq+'[EOS_K]')[1:-1]
                        #logging.info(self.tokenizer.decode(context_ids).replace(' ', ''))
                        max_len=len(context_ids)+10
                        eos_id=self.tokenizer.convert_tokens_to_ids('[EOS_SI]')
                        outputs = self.model.generate(input_ids=torch.tensor([context_ids]).to(self.model.device),
                                                pad_token_id=cfg.pad_id, max_length=max_len, eos_token_id=eos_id)
                        generated = outputs[0].cpu().numpy().tolist()
                        if eos_id not in generated:
                            generated[-1]=eos_id
                        SI=self.tokenizer.decode(generated[len(context_ids):-1]).replace(' ', '')
                        # generate system response
                        context_ids=generated
                        #logging.info(self.tokenizer.decode(context_ids).replace(' ', ''))
                        max_len=len(context_ids)+65
                        eos_id=self.tokenizer.convert_tokens_to_ids('[EOS_S]')
                        outputs = self.model.generate(input_ids=torch.tensor([context_ids]).to(self.model.device),
                                                pad_token_id=cfg.pad_id, max_length=max_len, eos_token_id=eos_id)
                        generated = outputs[0].cpu().numpy().tolist()
                        if eos_id not in generated:
                            generated[-1]=eos_id
                        resp=self.tokenizer.decode(generated[len(context_ids):-1]).replace(' ', '')
                        # delete repetition
                        repeated=re.findall(r'(.{3,})\1+', resp)
                        for p in repeated:
                            if p in resp:
                                idx=resp.index(p)+len(p)
                                resp=resp[:idx]
                        turn['history_ents']=EN_list_seq
                        turn['current_ents']=EN
                        turn['用户意图-生成']=UI
                        turn['查询结果']=KB_seq
                        turn['客服意图-生成']=SI
                        turn['客服-生成']=resp
                        if 'info' in turn:
                            turn.pop('info')
            logging.info('Dial num:{}, turn num:{}, testing time:{:.3f} min'.format(dial_num, turn_num, (time.time()-st)/60))
            json.dump(test_data, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        eval_result=eval_end_to_end(test_data)
        logging.info(eval_result)

    def test_context_to_resp(self, data='test'):
        encoded_data=read_data(self.tokenizer)
        test_data=encoded_data['test'] if data=='test' else encoded_data['dev']
        sep_id=self.tokenizer.convert_tokens_to_ids('[EOS_K]')
        eos_id=self.tokenizer.convert_tokens_to_ids('[EOS_S]')
        test_dataloader=DataLoader(test_data, batch_size=cfg.eval_batch_size, collate_fn=lambda x:test_collate_fn(x, sep_id), shuffle=False)
        self.model.eval()
        max_len=50
        gens, oracles, contexts = [], [], []
        st=time.time()
        with torch.no_grad():
            for batch in test_dataloader:
                # first predict the user intent
                # the generate 
                inputs, labels = batch[0], batch[1]
                gen_batch=self.generate_batch(self.model, inputs, max_len, eos_id)
                gens+=self.convert_batch_ids_to_tokens(self.tokenizer, gen_batch, eos_id)
                oracles+=self.convert_batch_ids_to_tokens(self.tokenizer, labels, eos_id)
                contexts+=self.convert_batch_ids_to_tokens(self.tokenizer, inputs, sep_id)
        logging.info('Generation time:{:.2f} min'.format((time.time()-st)/60))
        (P, R, F1), bleu = eval_context_to_response(gens, oracles)
        logging.info('Intent P/R/F1:{:.3f},{:.3f},{:.3f}, BLEU of response:{:.2f}'.format(P, R, F1, bleu))
        results=integrate_result(contexts, gens, oracles)
        json.dump(results, open(os.path.join(cfg.gpt_path, 'result.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    def gen_hidden_state(self, turn_batch, pv_batch, posterior=True, validate=False):
        cls = self.tokenizer.convert_tokens_to_ids('[CLS]')
        eos_entity_id = self.tokenizer.convert_tokens_to_ids('[EOS_E]')
        eos_b_id = self.tokenizer.convert_tokens_to_ids('[EOS_UI]')
        eos_db_id=self.tokenizer.convert_tokens_to_ids('[EOS_K]')
        eos_a_id=self.tokenizer.convert_tokens_to_ids('[EOS_SI]')
        sep_id=self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.model.eval()
        max_len_en=20
        max_len_b=30
        max_len_k=80
        max_len_a=10
        with torch.no_grad():
            # generate bspn
            if not cfg.db_change:
                if cfg.no_user_intent:
                    entity_gen = turn_batch['entity'] if 'entity' in turn_batch else turn_batch['user']
                else:
                    contexts = convert_eval_batch_turn(cls,turn_batch,pv_batch, mode='gen_ent', posterior=posterior)
                    entity_batch=self.generate_batch(self.model, contexts, max_len_en, eos_entity_id)
                    entity_gen = self.get_xspn(entity_batch,eos_entity_id,sep_id)

            #if cfg.ground_truth:
            #    contexts = convert_eval_batch_turn(cls,turn_batch,pv_batch, mode='gen_bspn', posterior=posterior,ent_gen=turn_batch['entity'])
            #else:
            if cfg.no_user_intent:
                bs_gen = turn_batch['bspn'] if 'bspn' in turn_batch else turn_batch['user']
            else:
                contexts = convert_eval_batch_turn(cls,turn_batch,pv_batch, mode='gen_bspn', posterior=posterior,ent_gen=entity_gen)
                bspn_batch=self.generate_batch(self.model, contexts, max_len_b, eos_b_id)
                bs_gen = self.get_xspn(bspn_batch,eos_b_id,sep_id)

            if cfg.ground_truth: # only gt_bspn for comparison
                contexts = convert_eval_batch_turn(cls,turn_batch,pv_batch, mode='gen_kb', 
                bspn_gen=turn_batch['bspn'],ent_gen=entity_gen, posterior=posterior) # turn_batch['entity']
            else:
                contexts = convert_eval_batch_turn(cls,turn_batch,pv_batch, mode='gen_kb', 
                bspn_gen=bs_gen,ent_gen=entity_gen, posterior=posterior)
            if cfg.gpt:
                db_batch=self.generate_batch(self.model, contexts, max_len_k, eos_db_id)
            else:
                inputs = {}
                inputs['contexts'] = contexts # not used yet, used in jsa
                inputs['contexts_np'],inputs['attention'] = padSeqs_gpt(contexts, cfg.pad_id, attention=True)
                inputs = self.add_torch_input(inputs)
                db_batch=self.model.generate(input_ids=inputs['contexts_tensor'], attention_mask=inputs['attention_tensor'], max_length=(max_len_k+max_len_a), eos_token_id=eos_a_id)    
            db_gen = self.get_xspn(db_batch, eos_db_id, sep_id)

            #if cfg.ground_truth:
            #    contexts = convert_eval_batch_turn(cls,turn_batch,pv_batch, mode='gen_ar', 
            #bspn_gen=turn_batch['bspn'],ent_gen=turn_batch['entity'], db_gen = turn_batch['db_new'], posterior=posterior)
            #else:
            contexts = convert_eval_batch_turn(cls,turn_batch,pv_batch, mode='gen_ar', 
                bspn_gen=bs_gen,ent_gen=entity_gen, db_gen = db_gen, posterior=posterior)
            if cfg.gpt:
                aspn_batch=self.generate_batch(self.model, contexts, max_len_a, eos_a_id)
                aspn_gen = self.get_xspn(aspn_batch, eos_a_id,sep_id) 
            else:
                aspn_gen = self.get_xspn(db_batch, eos_a_id,sep_id, sos_id=eos_db_id)

            if validate:# generate hidden states for validation
                turn_batch['查询结果'] = self.decode_batch((turn_batch['db_gtfg'] if cfg.fine_grained else turn_batch['db_gt'])) if cfg.gt_db else self.decode_batch(turn_batch['db']) #db_new
                turn_batch['用户意图'] = self.decode_batch(turn_batch['bspn'])
                turn_batch['客服意图'] = self.decode_batch(turn_batch['aspn'])
                turn_batch['用户意图-生成']=self.decode_batch(bs_gen)
                turn_batch['客服意图-生成']= self.decode_batch(aspn_gen)
                turn_batch['查询结果-生成'] = self.decode_batch(db_gen)
                turn_batch['客服'] = self.decode_batch(turn_batch['resp'])

            else:# generate hidden states for training
                turn_batch['bspn']=bs_gen
                turn_batch['entity']=entity_gen
                turn_batch['db']=db_gen
                turn_batch['db_gt']=db_gen
                turn_batch['db_new']=db_gen
                turn_batch['aspn']=aspn_gen
                if 'user_decode' not in turn_batch:
                    turn_batch['user_decode'] = self.decode_batch(turn_batch['user'])
                    turn_batch['resp_decode'] = self.decode_batch(turn_batch['resp'])
                    turn_batch['db_decode'] = self.decode_batch(turn_batch['db_gt'])
            #if validate:
            pv_batch = self.get_pv_batch(pv_batch, user = turn_batch['user'], resp = turn_batch['resp'], entity = entity_gen)
        return turn_batch, pv_batch 

    def get_pv_batch(self, pv_batch, user = None, resp = None, entity = None ):#user = None
        # only contain last_turn temporarily
        new_pv_batch=[] # pv_batch for next turn, return ent_list without any special tokens
        if pv_batch is None:# first turn
            for  u, r in zip( user, resp): 
                new_pv_batch.append(u + r) # remove eos_e
        else: 
            for hist, u, r in zip(pv_batch, user, resp):
                #ent_list = self.tokenizer.decode(hist).replace(' ', '').split(',')
                #turn_ent_list= self.tokenizer.decode(ent[:-1]).replace(' ', '').split(',')   # split ent by comma
                #for turn_ent in turn_ent_list:
                #    if turn_ent not in ent_list:
                #        ent_list.append(turn_ent)
                #ents = self.tokenizer.encode(','.join(ent_list))[1:-1]
                new_pv_batch.append(u + r)
        if cfg.db_change:
            new_pv_batch = resp
        return new_pv_batch

    def get_bspn(self,bs_tensor,eos_entity,eos_b_id,sep_id):
        
        if not isinstance(bs_tensor,list):
            bs_batch=bs_tensor.cpu().tolist()
        else:
            bs_batch=bs_tensor
        bs_gen=[]
        entity_gen=[]

        for i,bs in enumerate(bs_batch):
            if eos_entity in bs:
                entity = bs[:bs.index(eos_entity)+1]
            else:
                entity = [eos_entity]
            if eos_b_id in bs:
                bs = bs[:bs.index(eos_b_id)+1]
            else:
                bs = bs + [eos_b_id]
            if bs.count(eos_entity)>=1:
                last=bs[::-1].index(eos_entity)
                bs=bs[-last:]
            bs_new = []
            entity_new = []
            for token in bs:
                if token!=sep_id:
                    bs_new.append(token)
            for token in entity:
                if token!=sep_id:
                    entity_new.append(token)
            bs_gen.append(bs_new)
            entity_gen.append(entity_new)
        
        return bs_gen, entity_gen

    def get_xspn(self,input_tensor,eos_id,sep_id,sos_id = None):
        if not isinstance(input_tensor, list):
            input_batch=input_tensor.cpu().tolist()
        else:
            input_batch=input_tensor
        xspn_gen=[]
        for i ,xspn in enumerate(input_batch):
            if eos_id in xspn:
                xspn = xspn[:xspn.index(eos_id)+1]
            else:
                xspn[-1]=eos_id
            if sos_id:
                if sos_id in xspn:
                    xspn = xspn[xspn.index(sos_id)+1:] # multi sos_id not dealt with
                    if sos_id in xspn:
                        xspn = xspn[xspn.index(sos_id)+1:]
            xspn_new=[]
            for token in xspn:
                if token!=sep_id:
                    xspn_new.append(token)
            xspn_gen.append(xspn_new)
        return xspn_gen
    
    def query_KB(self,bspn_gen,KB, aspn_gen=None):
        KB_results=[]
        KB_seqs =[]
        for dial_num in range(len(bspn_gen)):
            UI = bspn_gen[dial_num]
            dial_KB = KB[dial_num]
            if aspn_gen:
                si = aspn_gen[dial_num]
            KB_result =[] 
            if '(' in UI:
                for intent in UI.split(','):
                    if '('  not in intent:
                        continue
                    act=intent[:intent.index('(')]
                    info=re.findall(r'\((.*?)\)', intent)
                    for e in info:
                        e=e.strip('-')
                        if '-' in e:
                            if len(e.split('-'))!=2:
                                continue
                            ent_name, prop=e.split('-')
                            res=query(dial_KB, ent_name=ent_name, prop=prop)
                        elif e.lower() in ['业务','数据业务','套餐', '主套餐','附加套餐','国际漫游业务','流量包','长途业务','4g套餐','5g套餐']:
                            res=query(dial_KB, ent_type=e)
                        else:
                            res=query(dial_KB, prop=e)
                        if res is not None:
                            if isinstance(res, list):
                                KB_result.append(','.join(res))
                            else:
                                KB_result.append(res)
            elif ('询问' in UI) or ('求助-查询' in UI) : # ('通知' in turn['客服意图'])
                if query(dial_KB):
                    KB_result.append(query(dial_KB))
            elif cfg.act_change : # ('通知' in turn['客服意图'])
                if '通知' in si:
                    if query(dial_KB) :
                        KB_result.append(query(dial_KB))
            KB_seq=','.join(KB_result)
            # generate system intent
            KB_results.append(self.tokenizer.encode(KB_seq+'[EOS_K]')[1:-1])
            KB_seqs.append(KB_seq)
        return KB_results,KB_seqs
    
    def query_KB_retrieval(self, model, tokenizer, spoken, context, KB, ebm=None):
        # threshold for ebm, can set higher than original
        threshold = cfg.threshold
        # threshold = 0.1
        KB_results=[]
        KB_seqs =[]
        model.eval()
        if ebm:
            ebm.eval()
        for dial_num in range(len(spoken)):
            c = context[dial_num]
            spoken_encoded = tokenizer.encode(c)
            s = spoken[dial_num]
            dial_KB = KB[dial_num]
            KB_result =[] 
            KB_query = []
            candidates = []
            for _,kvs in dial_KB.items():
                for k,v in  kvs.items():
                    if k !='type': # remove type for now
                        k_reg = k.replace('name','名称').replace('type','类型')
                        if (',' not in v) and (v!= ''):
                            if v not in s:
                                sv=(k_reg +':'+ v).lower()
                                KB_query.append({'context':spoken_encoded,'triple':tokenizer.encode(sv)})
                                candidates.append(sv)
                        else:
                            tmp_v = []
                            for value in v.split(','):
                                if value not in s:
                                    tmp_v.append(value)
                                    if cfg.fine_grained:
                                        sv=(k_reg +':'+ v).lower()
                                        KB_query.append({'context':spoken_encoded,'triple':tokenizer.encode(sv)})
                                        candidates.append(sv)
                            if not cfg.fine_grained:
                                if tmp_v != []:
                                    sv=(k_reg +':'+ ','.join(tmp_v)).lower()
                                    KB_query.append({'context':spoken_encoded,'triple':tokenizer.encode(sv)})   
                                    candidates.append(sv)
            if KB_query!=[]:
                # use bert model to get the relevent knowledge pieces
                if not cfg.rescore_for_generation:
                    batch = collate_fn(KB_query)
                    if cfg.only_one_model:
                        #candidate = copy.deepcopy(batch['input'])
                        for key, val in batch.items():
                            batch[key] = batch[key].to(cfg.device[0])
                        with torch.no_grad():
                            logits = model(input_ids=batch["input"], attention_mask=batch["input_attention"], token_type_ids=batch["input_type"]).logits
                            probs = F.softmax(logits, dim=1)
                            predicts=(probs[:,0]>threshold).cpu().tolist()
                        for num in range(len(predicts)):
                            if predicts[num]:
                                KB_result.append(candidates[num])
                # use ebm to get the rescore the results
                else:
                    topk_num = cfg.test_num
                    # change propose to topk1
                    p_batch = collate_fn(KB_query)
                    for key, val in p_batch.items():
                        if type(p_batch[key]) is list:
                            continue
                        p_batch[key] = p_batch[key].to(cfg.device[-1])
                    if p_batch!={}:
                        with torch.no_grad():
                            p_logits = model(input_ids=p_batch["input"], attention_mask=p_batch["input_attention"], token_type_ids=p_batch["input_type"]).logits
                            probs = F.softmax(p_logits, dim=1)
                            accept_prob = probs[:,0].cpu().tolist() # 0 means coherency in bert pretraining
                    else:
                        accept_prob = []
                    accept_result = []
                    triples_accepted = []
                    triples_accepted_idx = []
                    accepted_probs = 1.0
                    triples = []
                    triple_probs = []
                    triple_idxs = []
                    proposals = []
                    triple_num = len(accept_prob)
                    for num in range(triple_num):
                        # do not access gt_label in testing
                        #if KB_query[num]['label'] == 1:
                        #    gt.append(num)
                        if accept_prob[num] > threshold :
                            accept_result.append(num)
                        if accept_prob[num]> threshold:
                            triples_accepted.append(tokenizer.decode(KB_query[num]['triple']).replace('[CLS]','').replace('[SEP]','')) 
                            accepted_probs = accepted_probs*accept_prob[num] 
                            triples_accepted_idx.append(num)
                        else:   #if accept_prob[num]> reject_threshold:
                            triples.append(tokenizer.decode(KB_query[num]['triple']).replace('[CLS]','').replace('[SEP]',''))
                            triple_probs.append(accept_prob[num])
                            triple_idxs.append(num)
                    proposals = [(triples_accepted, accepted_probs, triples_accepted_idx)]
                    # topk_num, get_topk here by using beam search
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
                    # KB_query[0]['context']
                    input = get_retrieval_sequence(tokenizer, [c]*len(topk), ['；'.join(item[0]) for item in topk])
                    input.to(cfg.device[0])
                    #if cfg.add_extra_feature:
                    positive_count = torch.tensor([float(len(item[0])) for item in topk], dtype=torch.float).to(cfg.device[0])
                    with torch.no_grad():
                        if cfg.add_extra_feature:
                            logits = ebm(input_ids=input['input_ids'], attention_mask=input["attention_mask"], feature=positive_count).to('cpu').tolist()
                        else:
                            logits = ebm(input_ids=input['input_ids'], attention_mask=input["attention_mask"]).to('cpu').tolist()
                    if cfg.residual:
                        for j in range(len(logits)):
                            logits[j] = logits[j][0] + math.log(topk[j][1])
                    final = logits.index(max(logits))
                    KB_result = copy.deepcopy(proposals[final][0])
                    #triples = '；'.join(proposals[final][0]).lower()
            # fine_grained not support for rescoring
            if cfg.fine_grained:
                KB_dict = {}
                for k in KB_result:
                    slot = k.split(':')[0]
                    value = k.split(':')[1]
                    if slot not in KB_dict:
                        KB_dict[slot] = []
                    if value not in KB_dict[slot]: # need to be refined to ''.join(KB_dict[slot])
                        KB_dict[slot].append(value)
                KB_seq = ''
                for s,v in KB_dict.items():
                    KB_seq = KB_seq + s + ':' + ','.join(v) + '；'
                if '；' in KB_seq:
                    KB_seq = KB_seq[:-1]
            else:
                KB_seq=('；'.join(KB_result)).lower() # ','.join(KB_result)
            # generate system intent
            KB_results.append(self.tokenizer.encode(KB_seq+'[EOS_K]')[1:-1])
            KB_seqs.append(KB_seq)
        return KB_results,KB_seqs

    def decode_batch(self,result_encode,without_eos = True):
        result_decode =[]
        for encoded in result_encode:
            if without_eos:
                result_decode.append(self.tokenizer.decode(encoded[:-1]).replace(' ', '').replace('[CLS]', ''))
            else:
                result_decode.append(self.tokenizer.decode(encoded).replace(' ', '').replace('[CLS]', ''))
        return result_decode
    
    def get_spoken(self, spoken, new_input, role):
        result =[]
        hists = []
        role_shift = {'user':' 用户：', 'system':' 客服：'}
        for i in range(len(new_input)):
            s = ((spoken[i] if spoken!=[] else '') + role_shift[role] + new_input[i]).replace('[EOS_K]','').replace('[EOS_UI]','').replace('[EOS_SI]','').replace('[EOS_L]','').replace('[UNK]','')
            turns = s.split(' 用户：')
            if len(turns)> cfg.retrieve_hist:
                hist = ' 用户：' + (' 用户：').join(turns[-cfg.retrieve_hist:])
            else:
                hist = s
            result.append(s)
            hists.append(hist)
        return result, hists

    def encode_batch(self,batch,eos_id = None):
        result_encode =[]
        for sent in batch:
            if eos_id:
                result_encode.append(self.tokenizer.encode(sent)[1:-1] + [eos_id])
            else:
                result_encode.append(self.tokenizer.encode(sent)[1:-1])
        return result_encode

    def generate_batch_turn_level(self, batch, posterior=False, ground_truth_db = False):
        cls = self.tokenizer.convert_tokens_to_ids('[CLS]')
        eos_entity_id = self.tokenizer.convert_tokens_to_ids('[EOS_E]')
        eos_b_id = self.tokenizer.convert_tokens_to_ids('[EOS_UI]')
        eos_db_id=self.tokenizer.convert_tokens_to_ids('[EOS_K]')
        eos_a_id=self.tokenizer.convert_tokens_to_ids('[EOS_SI]')
        eos_r_id=self.tokenizer.convert_tokens_to_ids('[EOS_S]')
        sep_id=self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.model.eval()
        max_len_en=20
        max_len_b=30
        batch=transpose_batch(batch)
        max_len_a=20
        max_len_resp=90
        batch_size=len(batch[0]['user'])
        contexts=[[] for i in range(batch_size)]
        bs_gen=[]
        db_gen=[]
        resp_gen=[]
        entity_gen = []
        pv_batch=None
        spoken = []
        if cfg.retrieve_kb:
            bert_tokenizer = BertTokenizer.from_pretrained(cfg.bert_save_path)
            if cfg.only_one_model:
                if cfg.rescore_for_generation:
                    bert_model = BertForNextSentencePrediction.from_pretrained(cfg.bert_save_path)
                    ebm = EBM(cfg, bert_tokenizer)
                    save_path = os.path.join(cfg.ebm_save_path, 
                    f"model_allproposal{cfg.use_all_proposal}_mis_cache{cfg.train_ebm_mis}_residual{cfg.residual}"+ ("_add_feature" if cfg.add_extra_feature else "") + ".pt") #not joint training setting, add joint_train to change path
                    ebm.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
                    #EncoderModel(cfg,tokenizer)
                    bert_model.to(cfg.device[0])
                    ebm.to(cfg.device[-1])
                else:
                    bert_model = BertForNextSentencePrediction.from_pretrained(cfg.bert_save_path)#EncoderModel(cfg,tokenizer)
                    bert_model.to(cfg.device[0])
            else:
                bert_model = EncoderModel(cfg,bert_tokenizer)
                bert_model.to(cfg.device[0])
            bert_model.eval()
        with torch.no_grad():
            # generate bspn
            for turn_num, turn_batch in enumerate(batch):
                if cfg.db_change:
                    db_gen = turn_batch['db_all']
                    db_decode = self.decode_batch(db_gen)
                else:
                    contexts = convert_eval_batch_turn(cls, turn_batch, pv_batch, mode='gen_ent', posterior=posterior)
                    if not cfg.no_user_intent:
                        entity_batch=self.generate_batch(self.model, contexts, max_len_en, eos_entity_id)
                        entity_gen = self.get_xspn(entity_batch,eos_entity_id,sep_id)
                    else:
                        entity_gen = turn_batch['bspn'] # not use
                if cfg.db_change:
                    contexts = convert_eval_batch_turn(cls,turn_batch,pv_batch, mode='gen_bspn', posterior=posterior,db_gen=db_gen)
                else:
                    contexts = convert_eval_batch_turn(cls,turn_batch,pv_batch, mode='gen_bspn', posterior=posterior,ent_gen=entity_gen)
                turn_batch['用户'] = self.decode_batch(turn_batch['user'])
                spoken, hist = self.get_spoken(spoken, turn_batch['用户'], role='user')
                if not cfg.no_user_intent:
                    bspn_batch=self.generate_batch(self.model, contexts, max_len_b, eos_b_id)
                    bs_gen = self.get_xspn(bspn_batch,eos_b_id,sep_id)
                    bs_decode = self.decode_batch(bs_gen)
                    bs_gt = self.decode_batch(turn_batch['bspn'])
                else: # do not use or evaluate bspn, use ground-truth to avoid bug
                    bs_gen = turn_batch['bspn']
                    bs_decode = self.decode_batch(turn_batch['bspn'])
                    bs_gt = self.decode_batch(turn_batch['bspn'])

                if cfg.act_change:
                    contexts = convert_eval_batch_turn(cls,turn_batch,pv_batch, mode='gen_kb', posterior=posterior,bspn_gen=bs_gen,ent_gen=entity_gen)
                    aspn_batch=self.generate_batch(self.model, contexts, max_len_a, eos_a_id)
                    aspn_gen = self.get_xspn(aspn_batch,eos_a_id,sep_id)
                    as_decode = self.decode_batch(aspn_gen)
                if not cfg.db_change:    
                    if ground_truth_db:
                        db_gen = turn_batch['db_gt']
                        db_decode = self.decode_batch(db_gen)
                    else:
                        if cfg.act_change: # retrieval for this config is not supported yet
                            if cfg.ground_truth:
                                db_gen,db_decode = self.query_KB(bs_gt,turn_batch['KB'],as_decode)
                            else:
                                db_gen,db_decode = self.query_KB(bs_decode,turn_batch['KB'],as_decode)
                        else: 
                        # the current active mode
                            if cfg.rescore_for_generation:
                                db_gen,db_decode = self.query_KB_retrieval(bert_model, bert_tokenizer, spoken, hist, turn_batch['KB'], ebm)
                            elif cfg.retrieve_kb:
                                db_gen,db_decode = self.query_KB_retrieval(bert_model, bert_tokenizer, spoken, hist, turn_batch['KB'])
                            elif cfg.ground_truth:
                                db_gen,db_decode = self.query_KB(bs_gt,turn_batch['KB'])
                            else:
                                db_gen,db_decode = self.query_KB(bs_decode,turn_batch['KB']) 
                if not cfg.act_change:
                    contexts = convert_eval_batch_turn(cls,turn_batch,pv_batch, mode='gen_ar', 
                        bspn_gen=bs_gen,ent_gen=entity_gen, db_gen=db_gen, posterior=posterior)
                    if not cfg.gpt:
                        inputs = {}
                        inputs['contexts'] = contexts # not used yet, used in jsa
                        inputs['contexts_np'],inputs['attention'] = padSeqs_gpt(contexts, cfg.pad_id, attention=True)
                        inputs = self.add_torch_input(inputs)
                        aspn_batch=self.model.generate(input_ids=inputs['contexts_tensor'], attention_mask=inputs['attention_tensor'], max_length=max_len_a, eos_token_id=eos_a_id)#can be merged with response generation
                        # attention_mask=inputs['attention_tensor']
                        resp_batch=self.model.generate(input_ids=inputs['contexts_tensor'], attention_mask=inputs['attention_tensor'], max_length=max_len_resp, eos_token_id=eos_r_id)
                    else:
                        inputs,attentions = batch_align(contexts,return_attn=True)
                        inputs = torch.tensor(inputs).to(self.model.device)
                        attentions = torch.tensor(attentions).to(self.model.device)
                        aspn_batch=self.generate_batch(self.model, contexts, max_len_a, eos_a_id)
                        resp_batch=self.generate_batch(self.model, contexts, max_len_resp, eos_r_id)

                    aspn_gen = self.get_xspn(aspn_batch, eos_a_id,sep_id)
                    resp_gen = self.get_xspn(resp_batch, eos_r_id,sep_id,eos_a_id)
                    

                else:
                    contexts = convert_eval_batch_turn(cls,turn_batch,pv_batch, mode='gen_resp', 
                        bspn_gen=bs_gen,ent_gen=entity_gen, db_gen=db_gen, a_gen=aspn_gen, posterior=posterior)
                    resp_batch=self.generate_batch(self.model, contexts, max_len_resp, eos_r_id)
                    resp_gen = self.get_xspn(resp_batch, eos_r_id,sep_id,eos_a_id) 

                turn_batch['bspn_gen']=bs_gen
                if not cfg.db_change:
                    turn_batch['entity_gen']=entity_gen
                turn_batch['db_gen']=db_gen
                turn_batch['aspn_gen']=aspn_gen
                turn_batch['resp_gen']=resp_gen

                pv_batch = self.get_pv_batch(pv_batch, user = turn_batch['user'], resp = turn_batch['resp_gen'],  entity = entity_gen)

                turn_batch['用户意图-生成']=bs_decode
                turn_batch['查询结果'] = db_decode
                turn_batch['用户意图'] = self.decode_batch(turn_batch['bspn'])
                turn_batch['客服意图'] = self.decode_batch(turn_batch['aspn'])
                turn_batch['客服'] = self.decode_batch(turn_batch['resp'])
                turn_batch['客服意图-生成']= self.decode_batch(aspn_gen)
                turn_batch['客服-生成']=self.decode_batch(resp_gen)
                spoken, _=self.get_spoken(spoken, turn_batch['客服-生成'], role='system')

        return inverse_transpose_batch(batch)     
        
    def validate_fast(self, data='dev'):
        self.model.eval()
        print(cfg.retrieve_kb)
        encoded_data=read_data(self.tokenizer,self.posterior,return_dict=True)
        reserved_keys = ['用户意图-生成', '用户', '查询结果', '用户意图', '客服', '客服-生成', 
                '客服意图-生成', '客服意图-生成', 'bleu', 'success'] # reserver keys for demonstration
        if data == 'dev':
            eval_data = encoded_data['dev']
        elif data =='test':
            eval_data = encoded_data['test']
        origin_batch_size=cfg.batch_size
        cfg.batch_size=cfg.eval_batch_size
        batches=get_batches(eval_data,cfg.batch_size)#cfg.batch_size
        result_path=os.path.join(cfg.gpt_path,'result.json') if cfg.gpt else os.path.join(cfg.t5_path,'result.json')
        if cfg.mode == 'train_jsa':
            result_path=os.path.join(cfg.exp_path,'result.json')
        
        if os.path.exists(result_path) and cfg.mode=='test':
            results=json.load(open(result_path, 'r'))
            good_result = []
            new_result = []
            for dial in results:
                new_dial = []
                for tmp_turn in dial:
                    new_turn = {}
                    new_dial.append(new_turn)
                new_result.append(new_dial)
                bleu_score = sum(tmp['bleu'] for tmp in new_turn)/(len(new_turn) + 0.00001)
                if bleu_score>20:
                    good_result.append(new_dial)
            json.dump(good_result, open(result_path.replace('.json', '_good.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
            eval_result=eval_end_to_end(results)
            logging.info(eval_result)
            ui = eval_result['P/R/F1 for user intent'][2]
            si = eval_result['P/R/F1 for system intent'][2]
            bleu = eval_result['BLEU']
            success = eval_result['Success'] 
            eval_loss = bleu/50 + success # change eval to matchjsa-krtod paper
            logging.info(eval_loss)
            return eval_result
        else:
            result_collection = []
            st=time.time()
            for batch in batches:
                try:
                    if batch==[]:
                        continue
                    batch=self.generate_batch_turn_level(batch, ground_truth_db=cfg.gt_db)
                    for dialog in batch: 
                        result_collection.append(inverse_transpose_turn(dialog))
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        logging.info("WARNING: ran out of memory during validation and batch will be divided by half, batch size:{}, turn num:{}"\
                            .format(len(batch),len(batch[0])))
                        if hasattr(torch.cuda, 'empty_cache'):
                            with torch.cuda.device(self.device):
                                torch.cuda.empty_cache()
                        #divide the batch in half if out of memory
                        batches.insert(0,batch[:len(batch)//2])
                        batches.insert(1,batch[len(batch)//2:])
                    else:
                        logging.info(str(exception))
                        raise exception
            logging.info('Inference time:{:.3f} min'.format((time.time()-st)/60))
            eval_result, tmp_result = eval_end_to_end(result_collection, return_results=True)
            json.dump(tmp_result, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

            if cfg.mode=='test':
                new_result = []
                good_result = []
                for dial in tmp_result:
                    new_dial = []
                    for tmp_turn in dial:
                        new_turn = {}
                        for k in reserved_keys:
                            if k in tmp_turn:
                                new_turn[k] = tmp_turn[k]
                        new_dial.append(new_turn)
                    new_result.append(new_dial)
                    bleu_score = sum(tmp['bleu'] for tmp in new_dial)/len(new_turn)
                    if bleu_score>20:
                        good_result.append(new_dial)
                #json.dump(new_result, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
                json.dump(good_result, open(result_path.replace('.json', '_good.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
            logging.info(eval_result)
            cfg.batch_size = origin_batch_size
            return eval_result

    def validate_post(self, data='dev'):
        if cfg.mode == 'test_post':
            result_path=os.path.join(cfg.posterior_path,'result.json')
        else:
            result_path=os.path.join(cfg.exp_path,'result.json')
            if data == 'train':
                result_path=os.path.join(cfg.exp_path,'result_train.json')
        if os.path.exists(result_path) and cfg.mode=='test_post':
            results = json.load(open(result_path, 'r'))
            eval_result=eval_post(results)
            logging.info(eval_result)
            ui = eval_result['P/R/F1 for user intent'][2]
            si = eval_result['P/R/F1 for system intent'][2]
            db = eval_result['P/R/F1 for db prediction'][2] 
            eval_loss = ui + si + db
            logging.info(eval_loss)
            return eval_result
        
        else:
            self.model.eval()
            encoded_data=read_data(self.tokenizer,self.posterior,return_dict=True)
            if data == 'dev':
                eval_data = encoded_data['dev']
            if data == 'train':
                eval_data = encoded_data['train'][:1000]
            elif data =='test':
                eval_data = encoded_data['test']
            origin_batch_size=cfg.batch_size
            cfg.batch_size=cfg.eval_batch_size
            batches=get_batches(eval_data,cfg.batch_size)#cfg.batch_size
            result_collection = []
            st=time.time()
            for batch_idx, batch in enumerate(batches):
                pv_batch=None
                dial_batch=transpose_batch(batch)
                try:
                    for turn_num, turn_batch in enumerate(dial_batch):
                        turn_batch, pv_batch==self.gen_hidden_state(turn_batch, pv_batch,validate=True)
                    dial_batch=inverse_transpose_batch(dial_batch)
                    for dialog in dial_batch:
                        result_collection.append(inverse_transpose_turn(dialog))
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        logging.info("WARNING: ran out of memory during validation and batch will be divided by half, batch size:{}, turn num:{}"\
                            .format(len(batch),len(batch[0])))
                        if hasattr(torch.cuda, 'empty_cache'):
                            with torch.cuda.device(self.device):
                                torch.cuda.empty_cache()
                        #divide the batch in half if out of memory
                        batches.insert(0,batch[:len(batch)//2])
                        batches.insert(1,batch[len(batch)//2:])
                    else:
                        logging.info(str(exception))
                        raise exception
            logging.info('Inference time:{:.3f} min'.format((time.time()-st)/60))
            json.dump(result_collection, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
            eval_result=eval_post(result_collection)
            logging.info(eval_result)
            cfg.batch_size = origin_batch_size
            return eval_result

    def generate_pseudo_label(self):
        result_path='data/encoded_pseudo_data_new.json'
        encoded_file = os.path.join(cfg.data_dir, 'encoded_data_unl_whole.json')
        unl=json.load(open(encoded_file, 'r', encoding='utf-8'))
        #dials_unl=get_unsup(self.tokenizer)

        self.model.eval()
        origin_batch_size=cfg.batch_size
        cfg.batch_size=cfg.eval_batch_size
        batches=get_batches(unl,cfg.batch_size)#cfg.batch_size
        result_collection = []
        st=time.time()
        for batch in tqdm(batches):
            pv_batch=None
            dial_batch=transpose_batch(batch)
            try:
                for turn_num, turn_batch in enumerate(dial_batch):
                    turn_batch, pv_batch==self.gen_hidden_state(turn_batch, pv_batch)
                dial_batch=inverse_transpose_batch(dial_batch)
                for dialog in dial_batch:
                    result_collection.append(inverse_transpose_turn(dialog))
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    logging.info("WARNING: ran out of memory during validation and batch will be divided by half, batch size:{}, turn num:{}"\
                        .format(len(batch),len(batch[0])))
                    if hasattr(torch.cuda, 'empty_cache'):
                        with torch.cuda.device(self.device):
                            torch.cuda.empty_cache()
                    #divide the batch in half if out of memory
                    batches.insert(0,batch[:len(batch)//2])
                    batches.insert(1,batch[len(batch)//2:])
                else:
                    logging.info(str(exception))
                    raise exception
        logging.info('Inference time:{:.3f} min'.format((time.time()-st)/60))
        for dial in result_collection:
            for turn in dial:
                turn.pop('user_decode')
                turn.pop('resp_decode')
                turn.pop('db_decode')
                turn.pop('db_new')
                turn.pop('db')
        json.dump(result_collection, open(result_path, 'w', encoding='utf-8'), indent=2)
        return

    def generate_batch(self, model, contexts, max_len, eos_id, beam=1, return_prob=False):
        # generate by batch
        # contexts: a list of ids
        # max_len: the max generated length
        # eos_id: the end id
        # return: a batch of ids with pre pad 
        batch_size=len(contexts)
        # log prob of the generated text
        probs = [[0.0 for i in range(beam)] for j in range(batch_size)]
        end_flag=np.zeros(batch_size)
        if beam>1:
            beam_box=[beam]*batch_size
            beam_result=[[] for _ in range(batch_size)]
            max_prob=[-float('inf')]*batch_size
        past_key_values=None
        inputs,attentions=batch_align(contexts,left_len=max_len,return_attn=True)
        inputs=torch.tensor(inputs).to(model.device)
        attentions=torch.tensor(attentions).to(model.device)
        model.eval()
        with torch.no_grad():
            for i in range(max_len):
                if beam==1:
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                    if inputs.size(0)==0:
                        raise ValueError(contexts, inputs.cpu().list(), attentions)
                    outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)

                    past_key_values=outputs.past_key_values

                    preds=outputs.logits[:,-1,:].argmax(-1)#B

                    # calculating probs
                    case_probs = F.softmax(outputs.logits[:,-1,:]/cfg.temperature,dim=-1)
                    for num in range(preds.shape[0]):
                        index = preds.cpu().tolist()[num]
                        probs[num][0] = probs[num][0] + math.log(case_probs[num,index].cpu().item())

                    if i==0:
                        gen_tensor=preds.unsqueeze(1)
                    else:
                        gen_tensor=torch.cat([gen_tensor,preds.unsqueeze(1)], dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==eos_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                else:
                    if i==0:
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                                return_dict=True,use_cache=True,past_key_values=past_key_values)
                        past_key_values=[outputs.past_key_values]*beam
                        log_prob=F.log_softmax(outputs.logits[:, -1, :], -1) # B, V
                        beam_prob, beam_idx=torch.topk(log_prob, beam, -1) # B, beam
                        gen_tensor=beam_idx.unsqueeze(-1)# B, beam, 1
                        attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                        pv_beam_prob=beam_prob #B, beam
                        pv_beam_idx=beam_idx#B, beam
                    else:
                        for j in range(beam):
                            inputs=pv_beam_idx[:,j].unsqueeze(-1) # B, 1
                            outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                                return_dict=True,use_cache=True,past_key_values=past_key_values[j])
                            past_key_values[j]=outputs.past_key_values
                            log_prob=F.log_softmax(outputs.logits[:, -1, :], -1) # B, V
                            beam_prob, beam_idx=torch.topk(log_prob, beam, -1) # B, beam
                            if j==0:
                                prob_pool= beam_prob + pv_beam_prob[:, j].unsqueeze(-1).expand(-1, beam) # B, beam
                                id_pool=beam_idx
                            else:
                                prob_pool=torch.cat([prob_pool, beam_prob+pv_beam_prob[:, j].unsqueeze(-1).expand(-1, beam)], -1) # B, beam*beam
                                id_pool=torch.cat([id_pool, beam_idx], -1)# B, beam*beam
                        beam_prob, temp_id=torch.topk(prob_pool, beam, -1) #B, beam
                        beam_idx=torch.gather(id_pool, -1, temp_id)
                        temp_id=temp_id//beam
                        # past_key_values is a list of length beam, which
                        #past_key_values (Tuple[Tuple[torch.Tensor]] — Tuple of length config.n_layers, containing tuples of tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head)).

                        new_past_key_values=copy.deepcopy(past_key_values)
                        for b in range(batch_size):
                            gen_tensor[b, :, :]=gen_tensor[b, :, :].index_select(0, temp_id[b, :])
                            for t in range(beam):
                                for l in range(model.config.n_layer):
                                    for qk in range(2):
                                        new_past_key_values[t][l][qk][b, :, :, :]=past_key_values[temp_id[b, t]][l][qk][b, :, :, :]
                        past_key_values=new_past_key_values
                        #past_key_values=[past_key_values[t] for t in temp_id.cpu().list()]
                        gen_tensor=torch.cat([gen_tensor, beam_idx.unsqueeze(-1)],-1) #B, beam, T
                        attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                        pv_beam_prob=beam_prob #B, beam
                        pv_beam_idx=beam_idx
                    for m in range(batch_size):
                        for n, gen in enumerate(gen_tensor.cpu().tolist()[m]):
                            if (eos_id in gen or i==max_len-1) and len(beam_result[m])<beam:
                                #beam_box[m]-=1
                                avg_prob=copy.deepcopy(pv_beam_prob[m][n].item())
                                beam_result[m].append((gen, avg_prob))
                                pv_beam_prob[m][n]=-float('inf')
                    if all([len(tmp_result)==beam for tmp_result in beam_result]):
                        break  
                    #probs = beam_prob.cpu().item()

        if beam==1:
            if return_prob:
                return gen_tensor.cpu().tolist(), probs
            else:
                return gen_tensor.cpu().tolist()
        else:
            """
            for m in range(batch_size):
                for n, gen in enumerate(gen_tensor.cpu().tolist()[m]):
                    if eos_id in gen:
                        beam_box[m]-=1
                        # focus on the prob
                        avg_prob=pv_beam_prob[m][n].item() #len(gen)
                        beam_result[m].append((gen, avg_prob))
            """
                        #pv_beam_prob[m][n]=-float('inf')
            # we do not break during beam search
            #if not any(beam_box):
                #   break
            #for i, tup in enumerate(beam_result):
            #    beam_list=sorted(tup, key=lambda item:item[1], reverse=True)
            #    beam_result[i]=[item[0] for item in beam_list[:beam]]
            return beam_result

    def convert_batch_ids_to_tokens(self, tokenizer, input_ids, eos_id, return_ids=False):
        # input_ids: B*T
        # output: B*string
        outputs=[]
        outputs_ids=[]
        for sent_ids in input_ids:
            if eos_id in sent_ids:
                sent_ids=sent_ids[:sent_ids.index(eos_id)+1]
            else:
                sent_ids[-1]=eos_id
            outputs_ids.append(sent_ids)
            outputs.append(tokenizer.decode(sent_ids))
        if return_ids:
            return outputs, outputs_ids
        return outputs


def parse_arg_cfg(args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k == 'device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return cfg
    
class Semi_supervision(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device=cfg.device
        self.tokenizer = BertTokenizer.from_pretrained(cfg.posterior_path) # cfg.gpt_path
        self.cls = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep = self.tokenizer.convert_tokens_to_ids('[SEP]')
        if isinstance(self.device,list):
            self.device1 = self.device[0]
            self.device2 = self.device[1]
            self.device3 = self.device[-1]
            self.PrioriModel=Model(self.device1)#GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
            self.PosteriorModel=Model(self.device2,posterior=True)

        self.bert_tokenizer = BertTokenizer.from_pretrained(cfg.bert_save_path)
        if cfg.only_one_model:
            self.bert_model = BertForNextSentencePrediction.from_pretrained(cfg.bert_save_path)#EncoderModel(cfg,tokenizer)
            self.bert_model.to(self.device2)
        else:
            self.bert_model = EncoderModel(cfg,self.bert_tokenizer)
        # load ebm
        if cfg.jsa_ebm:            
            cfg.add_extra_feature = False
            # ebm
            cfg.use_all_proposal = True
            # mis settings
            cfg.train_ebm_mis = False # modified, MIS or not
            if cfg.train_ebm_mis:
                cfg.use_all_proposal = True # mis uses all proposals  
            cfg.residual = False # residual or not

            self.ebm = EBM(cfg, self.bert_tokenizer)
            if cfg.ebm_save_path == 'ret_exp/best_ebm':
                self.ebm_save_path = os.path.join(cfg.ebm_save_path, 
                f"model_allproposal{cfg.use_all_proposal}_mis_cache{cfg.train_ebm_mis}_residual{cfg.residual}"+ ("_add_feature" if cfg.add_extra_feature else "") + ".pt")
            else:
                self.ebm_save_path = cfg.ebm_save_path
            self.ebm.load_state_dict(torch.load(self.ebm_save_path, map_location=torch.device('cpu')))
            self.ebm.to(self.device3)
        self.bert_model.eval()

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
        #track nan in ebm training
        self.nan_num = 0 

    def jsa_train(self):
        cfg = self.cfg
        logging.info('------Running joint stochastic approximation------')
        # unlab_ratio = 1 # hyperparameter for num of jsa-training and supervised-training, type int, not use change the function 
        # in reader.get_unsup instead
        # use supervised sample multiple times if unlab_ratio is high, for example 4 and 9
        SUP_AUG = 3
        cfg.batch_size=cfg.batch_size*cfg.gradient_accumulation_steps #batch_size changed
        #encoded_data=read_data(self.tokenizer)
        #encoded_data_post=read_data(self.tokenizer,posterior=True)
        #train_data = encoded_data['train']
        #train_data_post = encoded_data_post['train']
        
        #dev_dataloader=DataLoader(encoded_data['dev'], batch_size=cfg.eval_batch_size, collate_fn=train_collate_fn)
        encoded_data=read_data(self.tokenizer,return_dict=True)
        train_data=encoded_data['train']
        random.shuffle(train_data)
        batches_lab = get_batches(train_data,cfg.batch_size)
        #num_dials=len(train_data)

        dials_unl=get_unsup(self.tokenizer,len(train_data))#list of unlabeled dialogs
        
        batches_unl = get_batches(dials_unl , batch_size = cfg.batch_size)
        logging.info('Labeled dials:{}, Unlabeled dials:{}'.format(len(train_data),len(dials_unl)))
        turn_nums = sum(len(dial) for dial in train_data) + sum(len(dial1) for dial1 in dials_unl)

        all_batches = []
        if cfg.debugging:
            batches_unl=batches_unl[9*len(batches_unl)//10:]
        for i in range(SUP_AUG):
            for batch in batches_lab:
                all_batches.append({'batch':transpose_batch(batch),'supervised':True})
        for batch in batches_unl:
            all_batches.append({'batch':transpose_batch(batch),'supervised':False})

        optimizer1, scheduler1 = self.PrioriModel.get_optimizers(turn_nums, self.PrioriModel.model) #num of total turns
        optimizer2, scheduler2 = self.PosteriorModel.get_optimizers(turn_nums, self.PosteriorModel.model)

        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info('  Num Batches = %d', len(all_batches))

        global_step = 0
        max_score=0
        min_loss=10000
        log_inputs = 3

        for epoch in range(cfg.epoch_num):
            GENERATE = True # if (epoch%2 == 0) else False # 2 can be modified to a larger number
            epoch_step = 0
            tr_loss, sup_loss, uns_loss = 0.0, 0.0, 0.0
            sup_step, uns_step=0, 0
            btm = time.time()
            self.PrioriModel.model.zero_grad()
            self.PosteriorModel.model.zero_grad()
            random.shuffle(all_batches)
            for batch_idx, dial_batch_dict in tqdm(enumerate(all_batches)):
                #unsup_train
                pv_batch=None
                spoken = []
                hist = []
                for turn_num, turn_batch in enumerate(dial_batch_dict['batch']):
                    if (not dial_batch_dict['supervised']) and GENERATE:
                        turn_batch, next_pv_batch=self.PosteriorModel.gen_hidden_state(turn_batch, pv_batch)
                    else:
                        next_pv_batch=self.PosteriorModel.get_pv_batch(pv_batch, user=turn_batch['user'], resp=turn_batch['resp'], entity=turn_batch['entity'])
                    first_turn = (turn_num == 0)
                    mini_batches, mini_pv_batches=split_turn_batch(turn_batch, cfg.origin_batch_size, other_batch=pv_batch)
                    if (not dial_batch_dict['supervised']) and GENERATE:
                        for i, batch in enumerate(mini_batches):
                            mini_pv_batch=None if turn_num==0 else mini_pv_batches[i]
                            inputs_prior, labels_prior = convert_batch_turn(self.cls,self.sep,batch, mini_pv_batch, first_turn, posterior=False)
                            inputs_posterior, labels_posterior = convert_batch_turn(self.cls,self.sep,batch, mini_pv_batch, first_turn, posterior=True)
                            self.PrioriModel.model.train()
                            self.PosteriorModel.model.train()

                            if len(spoken) < (i+1):
                                spoken.append([])
                                hist.append([])
                            spoken[i], hist[i]=self.get_spoken(spoken[i], batch['user_decode'], role='system')

                            if log_inputs > 0 :  # log inputs for the very first two turns
                                tmp_prior = self.tokenizer.decode(inputs_prior['contexts'][0])
                                tmp_posterior = self.tokenizer.decode(inputs_posterior['contexts'][0])
                                logging.info('Prior examples:\n{}'.format(tmp_prior))
                                logging.info("Posterior examples:\n{}".format(tmp_posterior))
                                #print(tmp_prior)
                                #print(tmp_posterior)
                                log_inputs -= 1

                            jsa_labels=(copy.deepcopy(inputs_posterior),copy.deepcopy(labels_posterior),copy.deepcopy(inputs_prior),copy.deepcopy(labels_prior),copy.deepcopy(batch['db_decode']))
                            # to tensor
                            inputs_prior = self.add_torch_input(inputs_prior)#B,T
                            inputs_posterior = self.add_torch_input(inputs_posterior,posterior=True)
                            labels_prior=self.add_torch_input(labels_prior)#B,T
                            labels_posterior=self.add_torch_input(labels_posterior,posterior=True)
                            # loss
                            with torch.no_grad():
                                outputs_prior=self.PrioriModel.model(inputs_prior['contexts_tensor'])
                                outputs_posterior=self.PosteriorModel.model(inputs_posterior['contexts_tensor'])#B,T,V
                                logits_pri=outputs_prior[0]
                                logits_post=outputs_posterior[0]
                            
                            #get prob
                            jsa_prob=self.get_jsa_prob(logits_pri,logits_post,\
                                    labels_prior['contexts_tensor'],labels_posterior['contexts_tensor'])
                            retrieval_prob = self.get_retrieval_prob(hist[i], batch['db_decode'])
                            for jsa_count in range(len(jsa_prob)):
                                jsa_prob[jsa_count] = jsa_prob[jsa_count] + retrieval_prob[jsa_count] 
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
                                with torch.no_grad():
                                    o_prior=self.PrioriModel.model(i_prior['contexts_tensor'])
                                    o_posterior=self.PosteriorModel.model(i_posterior['contexts_tensor'])#B,T,V
                                    lo_pri=o_prior[0]
                                    lo_post=o_posterior[0]
                                
                                #get prob
                                last_prob=self.get_jsa_prob(lo_pri,lo_post,\
                                        l_prior['contexts_tensor'],l_posterior['contexts_tensor'])
                                l_retrieval_prob = self.get_retrieval_prob(hist[i], temp_label[4])
                                for jsa_count in range(len(last_prob)):
                                    last_prob[jsa_count] = last_prob[jsa_count] + l_retrieval_prob[jsa_count]
                                #last_prob=copy.deepcopy(turn_batch['prob'][i])
                                #turn_batch['prob'][i]=jsa_prob
                                #last_prob=jsa_prob
                            
                            #update bspn
                            spoken[i], _=self.get_spoken(spoken[i], batch['resp_decode'], role='system') # need to be modified
                            for prob_num in range(min(len(jsa_prob),len(last_prob))):
                                if jsa_prob[prob_num]-last_prob[prob_num]>0:
                                    ratio=1.0
                                else:
                                    ratio=math.exp(jsa_prob[prob_num]-last_prob[prob_num])
                                if ratio<1.0:
                                    if random.random()>ratio:
                                        for j in range(5):
                                            if 'contexts_np' in jsa_labels[j]:
                                                jsa_labels[j].pop('contexts_np')
                                            if j!=4:
                                                jsa_labels[j]['contexts'][prob_num]=turn_batch['jsa_labels'][i][j]['contexts'][prob_num]
                                                #jsa_labels[j]['contexts_np'][prob_num]=dial_batch_dict['jsa_labels'][j]['contexts_np'][prob_num]
                                                jsa_labels[j]['lengths'][prob_num]=turn_batch['jsa_labels'][i][j]['lengths'][prob_num]
                                            else:
                                                jsa_labels[j][prob_num] = turn_batch['jsa_labels'][i][j][prob_num]
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
                                #outputs1=self.PrioriModel.model(inputs_prior['contexts_tensor'])    
                                #loss_pri=self.PrioriModel.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                                pass
                            else:
                                outputs1=self.PrioriModel.model(inputs_prior['contexts_tensor'])
                                loss_pri=self.PrioriModel.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                                outputs2=self.PosteriorModel.model(inputs_posterior['contexts_tensor'])
                                loss_pos=self.PosteriorModel.calculate_loss_and_accuracy(outputs2,labels_posterior['contexts_tensor'])
                            
                            # loss_pri.backward()
                            #if epoch<0:
                            if epoch!=0:
                                loss_pri.backward()
                                loss_pos.backward()
                                loss=loss_pri.item()+loss_pos.item()#.to(self.device1), .to(self.device2)
                                tr_loss += loss
                                uns_loss += loss
                            #else :
                            #    loss=loss_pri.item()
                            #    tr_loss += loss_pri.item()
                            #    uns_loss += loss_pri.item()
                                uns_step+=1
                    #supervised training
                    # comment: the mini batch settings can be modified
                    # the iteration is different from supervised learning process, 32 items in one iteration belong to the same turn
                    else:
                        if epoch!=0:
                            for i, batch in enumerate(mini_batches):
                                if dial_batch_dict['supervised']:
                                    mini_pv_batch=None if turn_num==0 else mini_pv_batches[i]
                                    inputs_prior, labels_prior = convert_batch_turn(self.cls, self.sep, batch, mini_pv_batch, first_turn, posterior=False)
                                    inputs_posterior, labels_posterior = convert_batch_turn(self.cls, self.sep, batch, mini_pv_batch, first_turn, posterior=True)
                                else:
                                    jsa_labels = copy.deepcopy(turn_batch['jsa_labels'][i])
                                    inputs_posterior = jsa_labels[0]
                                    labels_posterior = jsa_labels[1]
                                    inputs_prior = jsa_labels[2]
                                    labels_prior = jsa_labels[3]
                                inputs_prior = self.add_torch_input(inputs_prior)#B,T
                                labels_prior=self.add_torch_input(labels_prior)#B,T
                                inputs_posterior=self.add_torch_input(inputs_posterior,posterior=True)
                                labels_posterior=self.add_torch_input(labels_posterior,posterior=True)

                                outputs1 = self.PrioriModel.model(inputs_prior['contexts_tensor'])
                                loss_pri=self.PrioriModel.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                                loss_pri.backward()
                                if dial_batch_dict['supervised']:
                                    outputs2=self.PosteriorModel.model(inputs_posterior['contexts_tensor'])
                                    loss_pos=self.PosteriorModel.calculate_loss_and_accuracy(outputs2,labels_posterior['contexts_tensor'])
                                    loss_pos.backward()

                                # loss=loss_pri.to(self.device2)+loss_pos #.to(self.device1)
                                # loss.backward()
                                loss = loss_pri.item() + (loss_pos.item() if dial_batch_dict['supervised'] else 0)
                                tr_loss += loss
                                sup_loss +=loss
                                sup_step +=1
                    if epoch!=0:
                        torch.nn.utils.clip_grad_norm_(self.PrioriModel.model.parameters(), 5.0)
                        torch.nn.utils.clip_grad_norm_(self.PosteriorModel.model.parameters(), 5.0)
                        epoch_step+=1
                        optimizer1.step()
                        optimizer1.zero_grad()
                        optimizer2.step()
                        optimizer2.zero_grad()
                        global_step+=1
                        #if cfg.use_scheduler:
                        scheduler1.step()
                        scheduler2.step()
                        if self.tb_writer:
                            self.tb_writer.add_scalar('lr1', optimizer1.param_groups[0]["lr"],global_step)
                            self.tb_writer.add_scalar('lr2', optimizer2.param_groups[0]["lr"],global_step)
                            self.tb_writer.add_scalar('loss', loss, global_step)
                    pv_batch=next_pv_batch
                    #loss = 0
                    #loss_pri = 0
                    #loss_pos = 0  #clear loss to avoid out of memory
                        #torch.cuda.empty_cache()
                    
                    """
                    #sup_train, train a whole batch of 32 turns
                    if ratio_num == unlab_ratio-1:
                        train_loader = DataLoader(train_data[(batch_idx*cfg.batch_size):(batch_idx+1)*cfg.batch_size], batch_size=cfg.origin_batch_size, shuffle=True, collate_fn=train_collate_fn)
                        
                        train_post_loader = DataLoader(train_data_post[(batch_idx*cfg.batch_size):(batch_idx+1)*cfg.batch_size], batch_size=cfg.origin_batch_size, shuffle=True, collate_fn=train_collate_fn)

                        sup_loss = sup_loss + self.PrioriModel.train_one_step(train_loader,optimizer1, scheduler1)
                        sup_loss = sup_loss + self.PosteriorModel.train_one_step(train_post_loader,optimizer2, scheduler2)
                        sup_step = sup_step + 1
                    """
            logging.info('Epoch: {}, Train epoch time: {:.2f} min, loss:{:.3f}, avg_sup_loss:{:.3f}, avg_uns_loss:{:.3f}'.format(epoch, 
                (time.time()-btm)/60, tr_loss/(epoch_step+1e-10), sup_loss/(sup_step+1e-10), uns_loss/(uns_step+1e-10)))
            #eval_loss=self.PrioriModel.eval(dev_dataloader)
            if cfg.save_type =='max_score' and (epoch!=0):
                cfg.gt_db = False
                cfg.retrieve_kb = True
                eval_result=self.PrioriModel.validate_fast() # 'test'
                cfg.retrieve_kb = False
                cfg.gt_db = True
                ui = eval_result['P/R/F1 for user intent'][2]
                si = eval_result['P/R/F1 for system intent'][2]
                bleu = eval_result['BLEU']
                success = eval_result['Success']
                logging.info('user:{:.3f}, system:{:.3f} , bleu:{:.3f}, success:{:.3f}'.format(ui, si, bleu, success))
                eval_loss = bleu/50 + success
                logging.info('Epoch:{}, Train epoch time:{:.2f} min, epoch loss:{:.3f}, eval loss:{:.3f}'.format(epoch, (time.time()-btm)/60, tr_loss, eval_loss))
                self.tb_writer.add_scalar('eval_loss', eval_loss, epoch)
                if max_score < eval_loss:
                    max_score=eval_loss
                    self.PrioriModel.save_model()
                    self.PosteriorModel.save_model()
                else:
                    self.PrioriModel.save_model('last_model')
                    #self.PosteriorModel.save_model('last_model')
            elif cfg.save_type =='min_loss':
                if eval_loss<min_loss:
                    min_loss=eval_loss
                    self.PrioriModel.save_model()
                    self.PosteriorModel.save_model()

    # joint training energy-based model with generation model
    def jsa_train_ebm(self):
        cfg = self.cfg
        logging.info('------Running joint stochastic approximation with energy based model------')
        # unlab_ratio = 1 # hyperparameter for num of jsa-training and supervised-training, type int, not use change the function 
        # in reader.get_unsup instead
        # use supervised sample multiple times if unlab_ratio is high, for example 4 and 9
        cfg.batch_size=cfg.batch_size*cfg.gradient_accumulation_steps #batch_size changed
        #encoded_data=read_data(self.tokenizer)
        #encoded_data_post=read_data(self.tokenizer,posterior=True)
        #train_data = encoded_data['train']
        #train_data_post = encoded_data_post['train']
        
        #dev_dataloader=DataLoader(encoded_data['dev'], batch_size=cfg.eval_batch_size, collate_fn=train_collate_fn)
        encoded_data=read_data(self.tokenizer,return_dict=True)
        train_data=encoded_data['train']
        random.shuffle(train_data)
        batches_lab = get_batches(train_data,cfg.batch_size)
        #num_dials=len(train_data)

        dials_unl=get_unsup(self.tokenizer,len(train_data))#list of unlabeled dialogs
        
        batches_unl = get_batches(dials_unl , batch_size = cfg.batch_size)
        logging.info('Labeled dials:{}, Unlabeled dials:{}'.format(len(train_data),len(dials_unl)))
        turn_nums = sum(len(dial) for dial in train_data) + sum(len(dial1) for dial1 in dials_unl)

        all_batches = []
        
        SUP_AUG = 3 # control the num of sup-aug num

        #only aug sup data in first 4 epochs
        if cfg.jsa_unsup<4:
            EPOCH_TH = 3
        elif cfg.jsa_unsup==4:
            EPOCH_TH = 4 
        elif cfg.jsa_unsup==9:
            EPOCH_TH = 20
        #EBM_EPOCH_TH = 1 # train ebm after the first epochs
        if cfg.debugging:
            batches_unl=batches_unl[19*len(batches_unl)//20:]
            batches_lab=batches_lab[19*len(batches_lab)//20:]
        for i in range(SUP_AUG):
            for batch in batches_lab:
                all_batches.append({'batch':transpose_batch(batch),'supervised':True})
        for batch in batches_unl:
            all_batches.append({'batch':transpose_batch(batch),'supervised':False})

        optimizer1, scheduler1 = self.PrioriModel.get_optimizers(turn_nums, self.PrioriModel.model) #num of total turns
        optimizer2, scheduler2 = self.PosteriorModel.get_optimizers(turn_nums, self.PosteriorModel.model)
        #if cfg.jsa_ebm:
        # make the lr of ebm smaller to avoid nan
        optimizer3, scheduler3 = get_optimizers(num_samples=turn_nums ,model=self.ebm, lr=cfg.lr/10)

        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info('  Num Batches = %d', len(all_batches))

        global_step = 0
        max_score=0
        min_loss=10000
        log_inputs = 3

        for epoch in range(cfg.epoch_num):
            GENERATE = True # if (epoch%2 == 0) else False # 2 can be modified to a larger number
            if epoch>EPOCH_TH: # reset all_batches to reduce sup-num
                for batch in all_batches:
                    if batch['supervised']==True:
                        all_batches.remove(batch)
                for batch in batches_lab:
                    all_batches.append({'batch':transpose_batch(batch),'supervised':True})
            epoch_step = 0
            tr_loss, sup_loss, uns_loss, ebm_loss_label, ebm_loss_unlabel  = 0.0, 0.0, 0.0, 0.0, 0.0
            sup_step, uns_step=0, 0
            btm = time.time()
            self.PrioriModel.model.zero_grad()
            self.PosteriorModel.model.zero_grad()
            random.shuffle(all_batches)
            for batch_idx, dial_batch_dict in tqdm(enumerate(all_batches)):
                #unsup_train
                pv_batch=None
                spoken = []
                hist = []
                for turn_num, turn_batch in enumerate(dial_batch_dict['batch']):
                    if (not dial_batch_dict['supervised']) and GENERATE:
                        turn_batch, next_pv_batch=self.PosteriorModel.gen_hidden_state(turn_batch, pv_batch)
                    else:
                        next_pv_batch=self.PosteriorModel.get_pv_batch(pv_batch, user=turn_batch['user'], resp=turn_batch['resp'], entity=turn_batch['entity'])
                    first_turn = (turn_num == 0)
                    mini_batches, mini_pv_batches=split_turn_batch(turn_batch, cfg.origin_batch_size, other_batch=pv_batch)
                    if (not dial_batch_dict['supervised']) and GENERATE:
                        for i, batch in enumerate(mini_batches):
                            mini_pv_batch=None if turn_num==0 else mini_pv_batches[i]
                            inputs_prior, labels_prior = convert_batch_turn(self.cls,self.sep,batch, mini_pv_batch, first_turn, posterior=False)
                            inputs_posterior, labels_posterior = convert_batch_turn(self.cls,self.sep,batch, mini_pv_batch, first_turn, posterior=True)
                            self.PrioriModel.model.train()
                            self.PosteriorModel.model.train()

                            if len(spoken) < (i+1):
                                spoken.append([])
                                hist.append([])
                            spoken[i], hist[i]=self.get_spoken(spoken[i], batch['user_decode'], role='user')

                            if log_inputs > 0 :  # log inputs for the very first two turns
                                tmp_prior = self.tokenizer.decode(inputs_prior['contexts'][0])
                                tmp_posterior = self.tokenizer.decode(inputs_posterior['contexts'][0])
                                logging.info('Prior examples:\n{}'.format(tmp_prior))
                                logging.info("Posterior examples:\n{}".format(tmp_posterior))
                                #print(tmp_prior)
                                #print(tmp_posterior)
                                log_inputs -= 1

                            jsa_labels=(copy.deepcopy(inputs_posterior),copy.deepcopy(labels_posterior),copy.deepcopy(inputs_prior),copy.deepcopy(labels_prior),copy.deepcopy(batch['db_decode']))
                            # to tensor
                            inputs_prior = self.add_torch_input(inputs_prior)#B,T
                            inputs_posterior = self.add_torch_input(inputs_posterior,posterior=True)
                            labels_prior=self.add_torch_input(labels_prior)#B,T
                            labels_posterior=self.add_torch_input(labels_posterior,posterior=True)
                            # loss
                            with torch.no_grad():
                                outputs_prior=self.PrioriModel.model(inputs_prior['contexts_tensor'])
                                outputs_posterior=self.PosteriorModel.model(inputs_posterior['contexts_tensor'])#B,T,V
                                logits_pri=outputs_prior[0]
                                logits_post=outputs_posterior[0]
                            
                            #get prob
                            jsa_prob=self.get_jsa_prob(logits_pri,logits_post,\
                                    labels_prior['contexts_tensor'],labels_posterior['contexts_tensor'])
                            retrieval_prob = self.get_retrieval_prob_ebm(hist[i], batch['db_decode'])
                            for jsa_count in range(len(jsa_prob)):
                                jsa_prob[jsa_count] = jsa_prob[jsa_count] + retrieval_prob[jsa_count] 
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
                                with torch.no_grad():
                                    o_prior=self.PrioriModel.model(i_prior['contexts_tensor'])
                                    o_posterior=self.PosteriorModel.model(i_posterior['contexts_tensor'])#B,T,V
                                    lo_pri=o_prior[0]
                                    lo_post=o_posterior[0]
                                
                                #get prob
                                last_prob=self.get_jsa_prob(lo_pri,lo_post,\
                                        l_prior['contexts_tensor'],l_posterior['contexts_tensor'])
                                l_retrieval_prob = self.get_retrieval_prob_ebm(hist[i], temp_label[4])
                                for jsa_count in range(len(last_prob)):
                                    last_prob[jsa_count] = last_prob[jsa_count] + l_retrieval_prob[jsa_count]
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
                                        for j in range(5):
                                            if 'contexts_np' in jsa_labels[j]:
                                                jsa_labels[j].pop('contexts_np')
                                            if j!=4:
                                                jsa_labels[j]['contexts'][prob_num]=turn_batch['jsa_labels'][i][j]['contexts'][prob_num]
                                                #jsa_labels[j]['contexts_np'][prob_num]=dial_batch_dict['jsa_labels'][j]['contexts_np'][prob_num]
                                                jsa_labels[j]['lengths'][prob_num]=turn_batch['jsa_labels'][i][j]['lengths'][prob_num]
                                            else:
                                                jsa_labels[j][prob_num] = turn_batch['jsa_labels'][i][j][prob_num]
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
                                #outputs1=self.PrioriModel.model(inputs_prior['contexts_tensor'])    
                                #loss_pri=self.PrioriModel.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                                pass
                            else:
                                outputs1=self.PrioriModel.model(inputs_prior['contexts_tensor'])
                                loss_pri=self.PrioriModel.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                                outputs2=self.PosteriorModel.model(inputs_posterior['contexts_tensor'])
                                loss_pos=self.PosteriorModel.calculate_loss_and_accuracy(outputs2,labels_posterior['contexts_tensor'])

                            # loss_pri.backward()
                            #if epoch<0:
                            if epoch!=0:
                                # train_ebm
                                # sampling using posterior model
                                if cfg.jsa_ebm_joint:
                                    loss_ebm_un = self.train_ebm_unlabelled(hist[i], batch, mini_pv_batch, temp_label[4])
                                loss_pri.backward()
                                loss_pos.backward()
                                if cfg.jsa_ebm_joint:
                                    #loss_ebm.backward()
                                    ebm_loss_unlabel += loss_ebm_un
                                loss=loss_pri.item()+loss_pos.item()#.to(self.device1), .to(self.device2)
                                tr_loss += loss
                                uns_loss += loss
                                #else :
                                #    loss=loss_pri.item()
                                #    tr_loss += loss_pri.item()
                                #    uns_loss += loss_pri.item()
                                uns_step+=1
                            spoken[i], _=self.get_spoken(spoken[i], batch['resp_decode'], role='system') # need to be modified
                    #supervised training
                    # comment: the mini batch settings can be modified
                    # the iteration is different from supervised learning process, 32 items in one iteration belong to the same turn
                    else:
                        if epoch!=0:
                            for i, batch in enumerate(mini_batches):
                                if dial_batch_dict['supervised']:
                                    mini_pv_batch=None if turn_num==0 else mini_pv_batches[i]
                                    inputs_prior, labels_prior = convert_batch_turn(self.cls, self.sep, batch, mini_pv_batch, first_turn, posterior=False)
                                    inputs_posterior, labels_posterior = convert_batch_turn(self.cls, self.sep, batch, mini_pv_batch, first_turn, posterior=True)
                                else:
                                    jsa_labels = copy.deepcopy(turn_batch['jsa_labels'][i])
                                    inputs_posterior = jsa_labels[0]
                                    labels_posterior = jsa_labels[1]
                                    inputs_prior = jsa_labels[2]
                                    labels_prior = jsa_labels[3]
                                inputs_prior = self.add_torch_input(inputs_prior)#B,T
                                labels_prior=self.add_torch_input(labels_prior)#B,T
                                inputs_posterior=self.add_torch_input(inputs_posterior,posterior=True)
                                labels_posterior=self.add_torch_input(labels_posterior,posterior=True)

                                outputs1 = self.PrioriModel.model(inputs_prior['contexts_tensor'])
                                loss_pri=self.PrioriModel.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                                loss_pri.backward()
                                if dial_batch_dict['supervised']:
                                    outputs2=self.PosteriorModel.model(inputs_posterior['contexts_tensor'])
                                    loss_pos=self.PosteriorModel.calculate_loss_and_accuracy(outputs2,labels_posterior['contexts_tensor'])
                                    loss_pos.backward()

                                # loss=loss_pri.to(self.device2)+loss_pos #.to(self.device1)
                                # loss.backward()
                                # train the ebm
                                # get_context
                                user_decode = self.PrioriModel.decode_batch(batch['user'])
                                resp_decode = self.PrioriModel.decode_batch(batch['resp'])
                                if len(spoken) < (i+1):
                                    spoken.append([])
                                    hist.append([])
                                spoken[i], hist[i]=self.get_spoken(spoken[i],user_decode, role='user')
                                if cfg.jsa_ebm_joint:
                                    loss_ebm_label = self.train_ebm_labelled(spoken[i], hist[i], batch['KB'], resp_decode)
                                    ebm_loss_label += loss_ebm_label
                                # self.bert_model

                                #update spoken
                                spoken[i], _=self.get_spoken(spoken[i], resp_decode, role='system')

                                loss = loss_pri.item() + (loss_pos.item() if dial_batch_dict['supervised'] else 0)
                                tr_loss += loss
                                sup_loss +=loss
                                sup_step +=1
                    if epoch!=0:
                        torch.nn.utils.clip_grad_norm_(self.PrioriModel.model.parameters(), 5.0)
                        torch.nn.utils.clip_grad_norm_(self.PosteriorModel.model.parameters(), 5.0)
                        torch.nn.utils.clip_grad_norm_(self.ebm.parameters(), 5.0)
                        epoch_step+=1
                        optimizer1.step()
                        optimizer1.zero_grad()
                        optimizer2.step()
                        optimizer2.zero_grad()
                        global_step+=1
                        #if cfg.use_scheduler:
                        scheduler1.step()
                        scheduler2.step()
                        if cfg.jsa_ebm_joint:
                            optimizer3.step()
                            optimizer3.zero_grad()
                            scheduler3.step()
                        if self.tb_writer:
                            self.tb_writer.add_scalar('lr1', optimizer1.param_groups[0]["lr"],global_step)
                            self.tb_writer.add_scalar('lr2', optimizer2.param_groups[0]["lr"],global_step)
                            self.tb_writer.add_scalar('lr3', optimizer3.param_groups[0]["lr"],global_step)
                            self.tb_writer.add_scalar('loss', loss, global_step)
                    pv_batch=next_pv_batch
                    #loss = 0
                    #loss_pri = 0
                    #loss_pos = 0  #clear loss to avoid out of memory
                        #torch.cuda.empty_cache()
                    
                    """
                    #sup_train, train a whole batch of 32 turns
                    if ratio_num == unlab_ratio-1:
                        train_loader = DataLoader(train_data[(batch_idx*cfg.batch_size):(batch_idx+1)*cfg.batch_size], batch_size=cfg.origin_batch_size, shuffle=True, collate_fn=train_collate_fn)
                        
                        train_post_loader = DataLoader(train_data_post[(batch_idx*cfg.batch_size):(batch_idx+1)*cfg.batch_size], batch_size=cfg.origin_batch_size, shuffle=True, collate_fn=train_collate_fn)

                        sup_loss = sup_loss + self.PrioriModel.train_one_step(train_loader,optimizer1, scheduler1)
                        sup_loss = sup_loss + self.PosteriorModel.train_one_step(train_post_loader,optimizer2, scheduler2)
                        sup_step = sup_step + 1
                    """
            logging.info('Epoch: {}, Train epoch time: {:.2f} min, loss:{:.3f}, avg_sup_loss:{:.3f}, avg_uns_loss:{:.3f}, avg_ebm_loss_label:{:.3f}, avg_ebm_loss_unlabel:{:.3f}'.format(epoch, 
                (time.time()-btm)/60, tr_loss/(epoch_step+1e-10), sup_loss/(sup_step+1e-10), uns_loss/(uns_step+1e-10), ebm_loss_label/(epoch_step+1e-10),
                ebm_loss_unlabel/(epoch_step+1e-10)))
            #eval_loss=self.PrioriModel.eval(dev_dataloader)
            if cfg.save_type =='max_score' and (epoch!=0):
                cfg.gt_db = False
                cfg.retrieve_kb = True
                eval_result=self.PrioriModel.validate_fast('test') # 'test'
                cfg.retrieve_kb = False
                cfg.gt_db = True
                ui = eval_result['P/R/F1 for user intent'][2]
                si = eval_result['P/R/F1 for system intent'][2]
                bleu = eval_result['BLEU']
                success = eval_result['Success']
                logging.info('user:{:.3f}, system:{:.3f} , bleu:{:.3f}, success:{:.3f}'.format(ui, si, bleu, success))
                eval_loss = bleu/50 + success
                logging.info('Epoch:{}, Train epoch time:{:.2f} min, epoch loss:{:.3f}, eval loss:{:.3f}'.format(epoch, (time.time()-btm)/60, tr_loss, eval_loss))
                self.tb_writer.add_scalar('eval_loss', eval_loss, epoch)
                if max_score < eval_loss:
                    max_score=eval_loss
                    self.PrioriModel.save_model()
                    self.PosteriorModel.save_model()
                    self.tokenizer.save_pretrained(cfg.exp_path)
                    # save_ebm
                    if cfg.jsa_ebm_joint:
                        ebm_save_path = os.path.join(cfg.exp_path, 'best_ebm.pt')
                        torch.save(self.ebm.state_dict(), ebm_save_path)
                else:
                    self.PrioriModel.save_model('last_model')
                    #self.PosteriorModel.save_model('last_model')
            elif cfg.save_type =='min_loss':
                if eval_loss<min_loss:
                    min_loss=eval_loss
                    self.PrioriModel.save_model()
                    self.PosteriorModel.save_model()
                    ebm_save_path = os.path.join(cfg.exp_path, 'best_ebm.pt')
                    self.tokenizer.save_pretrained(cfg.exp_path)
                    torch.save(self.ebm.state_dict(), ebm_save_path)
    # discard
    """
    def get_spoken(self,spoken,new_input, role):
        result =[]
        role_shift = {'user':' 用户：', 'system':' 客服：'}
        for i in range(len(new_input)):
            result.append(((spoken[i] if spoken!=[] else '') + role_shift[role] + new_input[i]).replace('[EOS_L]',''))
        return result
    """

    def get_spoken(self, spoken, new_input, role):
        result =[]
        hists = []
        role_shift = {'user':' 用户：', 'system':' 客服：'}
        for i in range(len(new_input)):
            s = ((spoken[i] if spoken!=[] else '') + role_shift[role] + new_input[i]).replace('[EOS_K]','').replace('[EOS_UI]','').replace('[EOS_SI]','').replace('[EOS_L]','').replace('[UNK]','')
            turns = s.split(' 用户：')
            if len(turns)> cfg.retrieve_hist + 1:
                hist = ' 用户：' + (' 用户：').join(turns[-cfg.retrieve_hist:])
            else:
                hist = s
            result.append(s)
            hists.append(hist)
        return result, hists

    def get_retrieval_prob(self, context, KB):
        return_probs = []
        tokenizer = self.bert_tokenizer
        for dial_num in range(len(context)):
            prob = 1.0
            spoken = context[dial_num]
            spoken_encoded = tokenizer.encode(spoken.replace('[EOS_SI]','')) # .replace('[UNK]','')'
            dial_KB = KB[dial_num]
            KB_result =[] 
            KB_query = []
            candidates = []
            # avoid special tokens in KB
            flag = 0
            for token in special_tokens:
                if token in dial_KB:
                    flag=1
            if flag==0:
                for triple in dial_KB.split('；'):
                    if triple!='':
                        KB_query.append({'context':spoken_encoded,'triple':tokenizer.encode(triple)})
            else:
                prob = math.exp(-100)
            if KB_query!=[]:
                batch = collate_fn(KB_query)
                if cfg.only_one_model:
                    #candidate = copy.deepcopy(batch['input'])
                    for key, val in batch.items():
                        batch[key] = batch[key].to(self.device2)
                    with torch.no_grad():
                        logits = self.bert_model(input_ids=batch["input"], attention_mask=batch["input_attention"], token_type_ids=batch["input_type"]).logits
                        probs = F.softmax(logits, dim=1)
                        predicts=(probs[:,0]).cpu().tolist() # because 0 in the NSP task means consistency
                    for num in range(len(predicts)):
                        prob = prob * predicts[num]
            return_probs.append(copy.deepcopy(math.log(prob)))
        return return_probs

    def train_ebm_unlabelled(self, contexts, batch, pv_batch, kb_sequences):
        # beam size is the proposal number for training ebm
        BEAM_SIZE = 4
        resps = batch['resp_decode']
        #kb_tokens = batch['db_decode']
        return_probs = []
        tokenizer = self.bert_tokenizer
        loss_ebm = 0.0
        cls = self.tokenizer.convert_tokens_to_ids('[CLS]')
        eos_db_id=self.tokenizer.convert_tokens_to_ids('[EOS_K]')
        sep_id=self.tokenizer.convert_tokens_to_ids('[SEP]')
        max_len_k=80
        # convert_eval_batch_turn deals with a batch input
        gen_contexts = convert_eval_batch_turn(cls, batch, pv_batch, mode='gen_kb', posterior=True)
        #gen_contexts.to(self.device2)
        db_batches = self.PosteriorModel.generate_batch(self.PosteriorModel.model, gen_contexts, max_len_k, eos_db_id, beam=BEAM_SIZE, return_prob=True)
        gt_input = get_retrieval_sequence(tokenizer, contexts, kb_sequences)
        gt_input.to(self.device3)
        gt_logits = self.ebm(input_ids=gt_input['input_ids'], attention_mask=gt_input["attention_mask"])
        if (torch.isnan(gt_logits).any() or torch.isinf(gt_logits).any()) and self.nan_num<5:
            print(f"nan found in unsup training: {gt_logits}")
            print(contexts)
            print(kb_sequences)
            self.nan_num = self.nan_num + 1
            loss = 0.0
        else:
            loss = -sum(gt_logits)

        for dial_num in range(len(contexts)):
            # prob = 1.0
            # dialog history
            context = contexts[dial_num]
            db_batch = db_batches[dial_num]
            spoken_encoded = tokenizer.encode(context.replace('[EOS_SI]','')) # .replace('[UNK]','')'
            # avoid special tokens in KB
            # getting the samples ready for training ebm
            flag = 0
            #if kb_sequence!='': 

            # generating proposals and get their probabilities

            # gen_hidden
            #db_batch, db_probs = self.PosteriorModel.generate_batch(self.PosteriorModel, [gen_context], max_len_k, eos_db_id, beam=BEAM_SIZE, return_prob=True)
            if len(db_batch)>0:
                db_pregen =  [] # tmp[0] for tmp in db_batch
                db_logits =  [] # tmp[1] for tmp in db_batch
                for tmp_num in range(len(db_batch)):
                    if db_batch[tmp_num][1]>-20.0: # remove those with low sampling probs
                        db_pregen.append(db_batch[tmp_num][0])
                        db_logits.append(db_batch[tmp_num][1])
            if len(db_pregen)>0:
                db_gen = self.PosteriorModel.get_xspn(db_pregen, eos_db_id, sep_id)
                db_decode = self.PosteriorModel.decode_batch(db_gen)
                propose_input = get_retrieval_sequence(tokenizer, [context]*len(db_decode), db_decode)
                propose_input.to(self.device3)
                logits = self.ebm(input_ids=propose_input['input_ids'], attention_mask=propose_input["attention_mask"])
                # combine logits with probability to get the final loss
                is_ratio = []
                for j in range(len(db_logits)):
                    if (logits[j].item())<-100.0:
                        is_ratio.append(math.exp(-100.0))
                        print(f"large is ratio found in unsup training:{db_logits[j]},{logits[j].item()}")
                        #nan_num = nan_num + 1
                    elif (logits[j].item())>100.0:
                        is_ratio.append(math.exp(100.0))
                        print(f"large is ratio found in unsup training:{db_logits[j]},{logits[j].item()}")
                    else:
                        if cfg.residual:
                            is_ratio.append(math.exp (logits[j].item()))
                        else:
                            is_ratio.append(math.exp(logits[j].item()))
                            # math.exp(-db_logits[j] + logits[j].item()
                normalize = sum(is_ratio)
                if normalize>0.0:
                    for j in range(len(db_logits)):
                        loss = loss + is_ratio[j]*logits[j]/normalize
        if loss!=0.0:
            if (torch.isnan(loss).any() or torch.isinf(loss).any()) and self.nan_num<5:
                print(f"nan found in neg sample in unsup training: {logits}")
                print(contexts)
                print(kb_sequences)
                self.nan_num = self.nan_num + 1
                loss = 0.0
            else:
                loss.backward()
                loss_ebm += loss.item()
        
        return loss_ebm

    def train_ebm_labelled(self, spokens, contexts, KB, resps):
        tokenizer = self.bert_tokenizer
        loss = 0.0
        for dial_num in range(len(contexts)):
            # prob = 1.0
            spoken = spokens[dial_num]
            context = contexts[dial_num]
            resp = resps[dial_num]
            spoken_encoded = tokenizer.encode(spoken.replace('[EOS_SI]','')) # .replace('[UNK]','')'
            dial_KB = KB[dial_num]
            KB_result =[]
            # avoid special tokens in KB
            # getting the samples ready for training ebm
            # refer to get_retrieval_data(data, tokenizer, dial_ids=None, ebm=True) for the origin implementation
            # todo: make this section a function: get_turn_level_ebm
            negative = []
            positive = []
            flag = 0
            for _,kvs in dial_KB.items():
                for k,v in  kvs.items():
                    if k !='type': # remove type for now
                        if (',' not in v) and (v!= ''):
                            if (v not in spoken) and (v not in resp):
                                negative.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ v))
                            elif v in resp:
                                positive.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ v))
                                #if v in resp:
                                #    positive.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ v))
                                #else:
                                #    negative.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ v))
                            #elif v in resp:
                            #    positive.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ v))
                        else:
                            tmp_v = []
                            flag = 0
                            for value in v.split(','):
                                if (value not in spoken) and (value not in resp):
                                    tmp_v.append(value)
                                elif value in resp:
                                    tmp_v.append(value)
                                    flag = 1
                                #elif value in resp:
                                #    tmp_v.append(value)
                                #    flag = 1
                            if (flag == 0) and (tmp_v!=[]):
                                negative.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ ','.join(tmp_v)))   
                            elif flag == 1:
                                positive.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ ','.join(tmp_v)))
            #hist, hist_seq = get_dial_hist(hist, turn)
            #seq = tokenizer.encode(hist_seq)
            context_encoded = tokenizer.encode(context.replace('[EOS_SI]',''))
            turn_cases = []
            for p in positive:
                turn_cases.append({'context':context_encoded, 'triple':p, 'label':1})
            for n in negative:
                turn_cases.append({'context':context_encoded, 'triple':n, 'label':-1})
            if turn_cases!=[]:
                #turn_case = {'context':spoken, 'cases':turn_cases}
                #cases.append(turn_case)
                #try:  # avoid OOM
                #positive_count = []
                #for cases in batch['cases']:
                gt_seq = ''
                tmp_count = 0
                #for c in cases:
                for c in turn_cases:
                    if c['label']==1:
                        gt_seq = gt_seq + tokenizer.decode(c['triple']).replace('[CLS]','').replace('[SEP]','') + '；'
                        tmp_count += 1 
                if '；' in gt_seq:
                    gt_seq = gt_seq[:-1] 
                # training 
                gt_input = get_retrieval_sequence(tokenizer, [context], [gt_seq])
                gt_input.to(self.device3)
                logits = self.ebm(input_ids=gt_input['input_ids'], attention_mask=gt_input["attention_mask"])
                if (torch.isnan(logits).any() or torch.isinf(logits).any()) and self.nan_num<5:
                    print(f"nan found in sup training:{logits.item()}")
                    self.nan_num = self.nan_num + 1
                    break
                else:
                    loss = loss - sum(logits)
                p_batch = collate_fn(turn_cases)
                for key, val in p_batch.items():
                    if type(p_batch[key]) is list:
                        continue
                    p_batch[key] = p_batch[key].to(self.device2)
                if p_batch!={}:
                    with torch.no_grad():
                        org_logits = self.bert_model(input_ids=p_batch["input"], attention_mask=p_batch["input_attention"], token_type_ids=p_batch["input_type"]).logits
                        probs = F.softmax(org_logits, dim=1)
                        accept_prob = probs[:,0].cpu().tolist() # 0 means coherency in bert pretraining
                        gt_label = p_batch['label'].cpu().tolist() 
                else:
                    accept_prob = []
                triple_num = len(accept_prob)
                # sampling, mis mode not supported yet
                if cfg.train_ebm_mis and 'cached' in turn_cases[0]:
                    proposals = [turn_cases[0]['cached'][0]]
                    proposal_log_probs = [turn_cases[0]['cached'][1]]
                    proposal_wrong_num = [turn_cases[0]['cached'][2]]
                else:
                    proposals = []
                    proposal_log_probs = []
                    proposal_wrong_num = []
                # debugging only 
                #print(cfg.train_sample_num)
                for sample_num in range(cfg.train_sample_num):
                    p_prob = 0.0
                    proposal = []
                    proposal_id = []
                    for num in range(triple_num):
                        p = random.random()
                        if p< accept_prob[num]:
                            proposal.append(tokenizer.decode(turn_cases[num]['triple'][1:-1]).replace(' ','')) # can be unified to the .replace('[CLS]','').replace('[SEP]','')
                            # can directly concatenate all the triples to improve efficiency
                            p_prob += math.log(accept_prob[num])
                            proposal_id.append(0)
                        else:
                            p_prob += math.log(1-accept_prob[num])
                            proposal_id.append(1)
                    if proposal_id!=gt_label or cfg.use_all_proposal: #use cfg.use_all_proposal to include gt_label to be trained
                        proposals.append('；'.join(proposal))
                        proposal_log_probs.append(p_prob)
                        proposal_wrong_num.append(sum(gt_label[i]!=proposal_id[i] for i in range(len(gt_label))))
                # get IS_loss, avoiding OOM
                is_logits = []
                sample_num = len(proposals)
                positive_count = torch.tensor([float(len(item.split('；'))) for item in proposals], 
                dtype=torch.float).to(self.device3)
                if sample_num>0:
                    input = get_retrieval_sequence(tokenizer, [context]*sample_num, proposals)
                    input.to(self.device3)
                    if cfg.add_extra_feature:
                        is_logits.extend(self.ebm(input_ids=input['input_ids'], attention_mask=input["attention_mask"], feature=positive_count))
                    else:
                        is_logits.extend(self.ebm(input_ids=input['input_ids'], attention_mask=input["attention_mask"]))
                    is_ratio = []
                    for j in range(sample_num):
                        if (-proposal_log_probs[j] + is_logits[j].item())>100:
                            is_ratio.append(math.exp(100))
                            print(f"large is ratio found in sup training:{proposal_log_probs[j]},{is_logits[j].item()}")
                        if (-proposal_log_probs[j] + is_logits[j].item())<-100:
                            is_ratio.append(math.exp(-100))
                            print(f"large is ratio found:{proposal_log_probs[j]},{is_logits[j].item()}")
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
                        turn_cases[0]['cached']=(proposals[j], proposal_log_probs[j], proposal_wrong_num[j])
                        # save cached results
                    else:
                        normalize = sum(is_ratio)
                    if normalize>0.0:
                        if cfg.train_ebm_mis:    
                            for index, length in mis_results.items():
                                loss = loss + (length*is_logits[index])/normalize
                        else:
                            for j in range(sample_num):
                                if cfg.reject_control:
                                    if is_ratio[j]/normalize>0.003: # add reject control
                                        loss = loss + (is_ratio[j]* is_logits[j])/normalize
                                    else:
                                        if random.random()<is_ratio[j]*200/normalize:
                                            loss = loss + 0.005*is_logits[j]
                                else:
                                    loss = loss + is_ratio[j]*is_logits[j]/normalize
        if loss != 0.0:
            loss.backward()
            return loss.item()
        else:
            return 0.0

    def get_retrieval_prob_ebm(self, context, KB):
        return_probs = []
        tokenizer = self.bert_tokenizer
        for dial_num in range(len(context)):
            # prob = 1.0
            spoken = context[dial_num]
            spoken_encoded = tokenizer.encode(spoken.replace('[EOS_SI]','')) # .replace('[UNK]','')'
            dial_KB = KB[dial_num]
            KB_result =[] 
            KB_query = []
            candidates = []
            # avoid special tokens in KB
            flag = 0
            for token in special_tokens:
                if token in dial_KB:
                    flag=1
            if flag==0:
                input = tokenizer((spoken + '[SEP]' + dial_KB).replace('[UNK]','g'),
                 return_tensors="pt", padding=True, truncation=True, max_length=512)
                input.to(self.device3)
                with torch.no_grad():
                    ebm_logits = self.ebm(input_ids=input['input_ids'], attention_mask=input["attention_mask"])
                    return_probs.append(ebm_logits.cpu().tolist()[0][0]) 
                #tokenizer.encode((spoken + '[SEP]' + dial_KB).replace('[UNK]','g'))
                #for triple in dial_KB.split('；'):
                #    if triple!='':
                #        KB_query.append({'context':spoken_encoded,'triple':tokenizer.encode(triple)})
            else:
                #prob = math.exp(-100)
                return_probs.append(-300.0) # the num can be refined
            #if KB_query!=[]:
            #prob = math.exp(logits)
            """
            batch = collate_fn(KB_query)
            if cfg.only_one_model:
                #candidate = copy.deepcopy(batch['input'])
                for key, val in batch.items():
                    batch[key] = batch[key].to(self.device2)
                logits = self.bert_model(input_ids=batch["input"], attention_mask=batch["input_attention"], token_type_ids=batch["input_type"]).logits
                probs = F.softmax(logits, dim=1)
                predicts=(probs[:,0]).cpu().tolist() # because 0 in the NSP task means consistency
                for num in range(len(predicts)):
                    prob = prob * predicts[num]
            """
            #return_probs.append(copy.deepcopy(math.log(prob)))
        return return_probs

    def get_jsa_prob(self,logits_pri,logits_post,labels_pri,labels_post):
        # logits_pri:B,T1,V
        # logits_post:B,T2,V
        # labels_pri:B,T1. uspn's,bspn's label in prior sequence
        # labels_post:B,T2. bspn's label in posterior sequence
        # what labels do is to find the logits corresponding to bspn
        prob=[]
        for dial_idx in range(logits_pri.size(0)):
            label_pri=labels_pri[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist() #pad_id处为0，bspn为1
            label_post=labels_post[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist()
            h_len_post=len(label_post)-label_post[::-1].index(1)-label_post.index(1)
            h_len_pri=len(label_pri)-label_pri[::-1].index(1)-label_pri.index(1)
            idx1=label_pri.index(1)
            idx2=label_post.index(1)
            probs_pri=F.softmax(logits_pri[dial_idx, idx1:idx1+h_len_pri-1,:],dim=-1)
            probs_post=F.softmax(logits_post[dial_idx, idx2:idx2+h_len_post-1,:],dim=-1)
            up=torch.tensor(0.0)
            down=torch.tensor(0.0)
        
            for up_num in range(probs_pri.size()[0]):#loc2-loc1-1
                #if probs_pri.size()[0]!=loc2-loc1-1
                #    print(probs_pri.size()[0])
                #    print(loc2-loc1-1)
                if probs_pri[up_num,labels_pri[dial_idx,idx1+up_num+1]]!=0:
                    up=up+math.log(probs_pri[up_num,labels_pri[dial_idx,idx1+up_num+1]])#probs_pri[up_num,:].max()
                else:
                    up=up-1000
            for down_num in range(probs_post.size()[0]):#loc4-loc3-1
                if probs_post[down_num,labels_post[dial_idx,idx2+down_num+1]]!=0:
                    down=down+math.log(probs_post[down_num,labels_post[dial_idx,idx2+down_num+1]])#probs_pri[down_num,labels_pri[logits_pri.size(1)-loc2+up_num]]
                else:
                    down=down-1000
            prob.append(up.item()-down.item())
        return prob    

    def add_torch_input(self, inputs, posterior=False):
        # to tensor and to device
        if 'contexts_np' not in inputs:
            inputs['contexts_np'],_=padSeqs_gpt(inputs['contexts'], cfg.pad_id)
        contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
        if posterior:
            contexts_tensor = contexts_tensor.to(self.device2)
        else:
            contexts_tensor = contexts_tensor.to(self.device1)
        inputs['contexts_tensor'] = contexts_tensor
        return inputs   

def main():
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')
    if not os.path.exists('./log'):
        os.mkdir('./log')

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()
    cfg.mode = args.mode
    parse_arg_cfg(args)
    if cfg.exp_path=='':
        experiments_path = './experiments'
        cfg.exp_path = os.path.join(experiments_path, cfg.exp_name)
        if not os.path.exists(cfg.exp_path):
            os.mkdir(cfg.exp_path)

    cfg._init_logging_handler()

    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    # initialize model
    m = Model(cfg.device)
    # train
    if cfg.mode=='train':
        m.train()
    if cfg.mode=='pretrain':
        m.train()
    if cfg.mode=='train_post':
        m = Model(cfg.device,posterior=True)
        m.train()
    if cfg.mode=='train_jsa':
        semi = Semi_supervision(cfg)
        semi.jsa_train()
    if cfg.mode=='train_jsa_ebm':
        semi = Semi_supervision(cfg)
        semi.jsa_train_ebm()
    if cfg.mode=='test':
        m.validate_fast(data = 'test')
        #m.test_end_to_end()
    if cfg.mode=='test_post':
        m = Model(cfg.device,posterior=True)
        m.validate_post(data = 'test')
    if cfg.mode=='generate_post': # generate pseudo label
        m = Model(cfg.device,posterior=True)
        m.generate_pseudo_label()

 
if __name__ == "__main__":
    main()
