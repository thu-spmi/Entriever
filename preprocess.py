import json
import random
import copy
import re
import collections
def data_statistics():
    data=json.load(open('data/seretod/dev_data.json', 'r', encoding='utf-8'))#'data/extra/data_label_v1.1.json'
    c1, c2 = 0, 0
    c3=0
    dials1, dials2, dials3 = [], [], []
    print(len(data))
    for dial in data:
        user_first=0
        service_first=0
        confusion=0
        speakers=set()
        for turn in dial['content']:
            temp=list(turn.keys())
            speakers.add(temp[0])
            speakers.add(temp[1])
            if '客服意图' in temp and '用户意图' in temp:
                if '意图混乱' in temp or '意图混乱' in turn['用户意图'] or '意图混乱' in turn['客服意图']:
                    confusion=1
                    continue
                if temp.index('客服意图')<temp.index('用户意图'):#客服在前
                    c1+=1
                    service_first=1
                else:
                    c2+=1
                    user_first=1
            else:
                c3+=1
                confusion=1
        if confusion:
            dials3.append(dial)
        elif user_first: # 对话中至少有一轮用户在前
            if len(speakers)==2 and not service_first:
                dials2.append(dial)
            else:
                dials3.append(dial)
        elif service_first: # 对话中至少有一轮客服在前
            if len(speakers)==2 and not user_first:
                dials1.append(dial)
            else:
                dials3.append(dial)
        else:
            dials3.append(dial)
    print('轮次——客服在前:', c1, '用户在前:', c2, '其他情况:', c3)
    print('对话——客服在前:', len(dials1), '用户在前:', len(dials2), '其他情况:', len(dials3))
    json.dump(dials1, open('data/service_first_dev.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dials2, open('data/user_first_dev.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dials3, open('data/others.json_dev', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

def data_statistics0():
    data=json.load(open('data/Raw_data.json', 'r', encoding='utf-8'))#
    c1, c2 = 0, 0
    c3=0
    dials1, dials2, dials3 = [], [], []
    dials_std=[]
    print(len(data))
    for dial in data:
        user_first=0
        service_first=0
        other=0
        confusion=0
        speakers=set()
        for turn in dial:
            temp=list(turn.keys())
            service, user=temp[:2]
            speakers.add(service)
            speakers.add(user)
            if '客服意图' in temp and '用户意图' in temp:
                if '意图混乱' in temp or '意图混乱' in turn['用户意图'] or '意图混乱' in turn['客服意图']:
                    confusion=1
                    continue
                if temp.index('客服意图')<temp.index('用户意图'):#客服在前
                    c1+=1
                    service_first=1
                else:
                    c2+=1
                    user_first=1
            else:
                c3+=1
                other=1
        if other or confusion:
            dials3.append(dial)
        elif user_first: # 对话中至少有一轮用户在前
            if len(speakers)==2 and not service_first:
                dials2.append(dial)
        else:
            dials1.append(dial)
            if len(speakers)==2:
                dials_std.append(dial)
    print('轮次——客服在前:', c1, '用户在前:', c2, '其他情况:', c3)
    print('对话——客服在前:', len(dials1), '用户在前:', len(dials2), '其他情况:', len(dials3))
    print('标准对话数:', len(dials_std))
    json.dump(dials1, open('data/part1.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dials2, open('data/part2.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dials3, open('data/part3.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dials_std, open('data/std_data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

def clear_data(data):
    dial_count, turn_count=0,0
    new_data=[]
    remaining_data=[]
    for d, dial in enumerate(data):
        dial_key_mission=0
        for t, turn in enumerate(dial):
            turn_key_mission=0
            if "info" in turn:
                if 'ents' not in turn['info']:
                    turn['info']['ents']=[]
                if 'triples' not in turn['info']:
                    turn['info']['triples']=[]
                for ent in turn['info']['ents']:
                    for key in ['name', 'id', 'type', 'pos']:
                        if key not in ent:
                            #print('Entities of dial {} turn {} has no {}'.format(d, t, key))
                            if 'ent-'+key in ent:
                                ent[key]=ent.pop('ent-'+key)
                            else:
                                #print('Entities of dial {} turn {} has no {}'.format(d, t, key))
                                if 'missing' not in ent:
                                    ent['missing']=key
                                else:
                                    ent['missing']+=','+key
                                dial_key_mission=1
                                turn_key_mission=1
                for triple in turn['info']['triples']:
                    for key in ['ent-id', 'ent-name', 'prop', 'value']:
                        if key not in triple:
                            #print('Triples of dial {} turn {} has no {}'.format(d, t, key))
                            if key=='ent-id' and 'id' in triple:
                                triple[key]=triple.pop('id')
                                #print('Triples of dial {} turn {} has {}'.format(d, t, 'id'))
                            elif key=='ent-name' and 'name' in triple:
                                triple[key]=triple.pop('name')
                                #print('Triples of dial {} turn {} has {}'.format(d, t, 'name'))
                            else:
                                #print('Triples of dial {} turn {} has no {}'.format(d, t, key))
                                if 'missing' not in triple:
                                    triple['missing']=key
                                else:
                                    triple['missing']+=','+key
                                dial_key_mission=1
                                turn_key_mission=1
                    if 'ref' in triple:
                        triple.pop('ref')
            turn_count+=turn_key_mission
        dial_count+=dial_key_mission
        if not dial_key_mission:
            new_data.append(dial)
        else:
            remaining_data.append(dial)
    print('Dials with missing keys:{}, turns with missing keys:{}'.format(dial_count, turn_count))
    return new_data, remaining_data

def restructure():
    # restructure the data so that user speaks first at all turns
    data=json.load(open('data/service_first_dev.json', 'r', encoding='utf-8'))
    data1=json.load(open('data/user_first_dev.json', 'r', encoding='utf-8'))
    #data, _=clear_data(data)
    #data1,_=clear_data(data1)
    new_data=[]
    missing_pos=0
    for dial in data:
        service, user=list(dial['content'][0].keys())[:2]
        new_item={'id':dial['id']}
        new_dial=[]
        for n in range(len(dial['content'])-1):
            turn1=dial['content'][n]
            turn2=dial['content'][n+1]
            new_turn={}
            new_turn['用户']=turn1[user].replace('[UNK]', '')
            new_turn['客服']=turn2[service].replace('[UNK]', '')
            new_turn['用户意图']=turn1['用户意图']
            new_turn['客服意图']=turn2['客服意图']
            new_turn.update({"info": {
                "ents":[],
			    "triples":[]
                }
            })
            if 'info' in turn1:
                for ent in turn1['info']['ents']:
                    pos=[]
                    if 'pos' not in ent:
                        continue
                    for p in ent['pos']:
                        if p==[]:
                            continue
                        if p[0]==2:
                            pt=copy.deepcopy(p)
                            pt[0]=1
                            pos.append(pt)
                    if len(pos)>0:
                        new_ent=copy.deepcopy(ent)
                        new_ent['pos']=pos
                        new_turn['info']['ents'].append(new_ent)
                
                if 'triples' in turn1['info']:
                    for triple in turn1['info']['triples']:
                        if 'value' not in triple:
                            continue
                        if triple['value'] in turn1[user]:
                            new_turn['info']['triples'].append(triple)
            if 'info' in turn2:
                for ent in turn2['info']['ents']:
                    pos=[]
                    if 'pos' not in ent:
                        continue
                    for p in ent['pos']:
                        if p==[]:
                            missing_pos+=1
                            continue
                        if p[0]==1:
                            pt=copy.deepcopy(p)
                            pt[0]=2
                            pos.append(pt)
                    if len(pos)>0:
                        new_ent=copy.deepcopy(ent)
                        new_ent['pos']=pos
                        new_turn['info']['ents'].append(new_ent)
                if 'triples' in turn2['info']:
                    for triple in turn2['info']['triples']:
                        if 'value' not in triple:
                            continue
                        if triple['value'] in turn2[service]:
                            new_turn['info']['triples'].append(triple)
            new_dial.append(new_turn)
        new_item['content']=new_dial
        new_data.append(new_item)
    
    for dial in data1:
        user, service=list(dial['content'][0].keys())[:2]
        new_item={'id':dial['id']}
        new_dial=[]
        for turn in dial['content']:
            turn['用户']=turn.pop(user).replace('[UNK]', '')
            turn['客服']=turn.pop(service).replace('[UNK]', '')
            new_dial.append(turn)
        new_item['content']=new_dial
        new_data.append(new_item)
    print('Total restructured data:', len(new_data))
    json.dump(new_data, open('data/restructured_data_dev.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

def extract_local_KB():
    data=json.load(open('data/restructured_data_dev.json', 'r', encoding='utf-8'))
    new_data=[]
    count=0
    turn_num=0
    query_num=0
    query_dial=0
    for n, dial in enumerate(data):
        entry={
            'id':dial['id'],
            'KB':{},
            'goal':{},
            'content':dial['content']
            }
        KB={}
        goal={}
        with_query=0
        # extract db
        for turn in dial['content']:
            turn_num+=1
            turn['用户意图']=turn['用户意图'].replace('（', '(').replace('）',')').replace('，',',')
            turn['客服意图']=turn['客服意图'].replace('（', '(').replace('）',')').replace('，',',')
            if '(' in turn['用户意图']:
                query_num+=1
                with_query=1
            if 'info' in turn:
                for ent in turn['info']['ents']:
                    if 'id' not in ent:
                        continue
                    if ent['name'] in turn['用户']:#只有用户提到的才加到goal中
                        if ent['id'] not in goal:
                            goal[ent['id']]={'type':ent['type']}
                        if 'name' not in goal[ent['id']]:
                            goal[ent['id']]['name']=set([ent['name']])
                        else:
                            goal[ent['id']]['name'].add(ent['name'])

                    if ent['id'] not in KB:
                        KB[ent['id']]={
                            'name':set([ent['name'].lower()]),
                            'type':ent.get('type', ' ').lower()
                        }
                    else:# we accumulate all the names for one entity
                        KB[ent['id']]['name'].add(ent['name'].lower())
                for triple in turn['info']['triples']:
                    if 'ent-id' not in triple or 'prop' not in triple:
                        continue
                    if triple['value'] in turn['用户']:
                        if triple['ent-id'].startswith('ent') and triple['ent-id'] not in goal:
                            goal[triple['ent-id']]={'name':set([triple['ent-name']])}
                        if triple['ent-id'] not in goal:   
                            goal[triple['ent-id']]={}
                        if triple['prop'] not in goal[triple['ent-id']]:
                            goal[triple['ent-id']][triple['prop']]=set([triple['value']])
                        else:
                            goal[triple['ent-id']][triple['prop']].add(triple['value'])

                    if triple['ent-id'].startswith('ent') and triple['ent-id'] not in KB:
                        #print('Triple appeare before entity:', triple['ent-id'])
                        KB[triple['ent-id']]={'name':set([triple['ent-name'].lower()])}
                        count+=1
                    if triple['ent-id'] not in KB:   
                        KB[triple['ent-id']]={}
                    if triple['prop'] not in KB[triple['ent-id']]:
                        KB[triple['ent-id']][triple['prop'].lower()]=set([triple['value']])
                    else:
                        KB[triple['ent-id']][triple['prop'].lower()].add(triple['value'])
            
            ui=turn['用户意图'] #user intent
            if '(' in ui:
                for intent in ui.split(','):
                    if '(' in intent:
                        act=intent[:intent.index('(')]
                        info=re.findall(r'\((.*?)\)', intent)
                        for e in info:
                            e=e.strip('(').strip(')')
                            if e in ['业务','数据业务','套餐', '主套餐','附加套餐','国际漫游业务','流量包','长途业务','4G套餐','5G套餐']:
                                item=act+'-'+e
                                if '咨询' not in goal:
                                    goal['咨询']=[item]
                                elif item not in goal['咨询']:
                                    goal['咨询'].append(item)
                            elif '-' in e:
                                ent_id=e[:5]
                                prop=e[5:].strip('-')
                                if ent_id in goal:
                                    if prop in goal[ent_id]:
                                        goal[ent_id][prop].add('?')
                                    else:
                                        goal[ent_id][prop]=set(['?'])
                                    if '用户意图' in goal[ent_id]:
                                        goal[ent_id]['用户意图'].add(act)
                                    else:
                                        goal[ent_id]['用户意图']=set([act])
                                else:
                                    goal[ent_id]={prop:set(['?']), '用户意图':set([act])}
                                
                            else: #查询个人信息
                                if 'NA' in goal:
                                    if e in goal['NA']:
                                        goal['NA'][e].add('?')
                                    else:
                                        goal['NA'][e]=set(['?'])
                                    if '意图' in goal['NA']:
                                        goal['NA']['意图'].add(act)
                                    else:
                                        goal['NA']['意图']=set([act])
                                else:
                                    goal['NA']={e:set(['?']), '意图':set([act])}
        if with_query:
            query_dial+=1
        for id, ent in KB.items():
            for key, value in ent.items():
                if isinstance(value, set):
                    KB[id][key]=','.join(list(value))
        for id, ent in goal.items():
            if isinstance(ent, list):
                goal[id]=','.join(ent)
            else:
                for key, value in ent.items():
                    if isinstance(value, set):
                        goal[id][key]=','.join(list(value))
        entry['KB']=KB
        entry['goal']=goal
        new_data.append(entry)
    print('Triple appeare before entity:', count)
    print('Total turns:', turn_num, 'query turns:', query_num)
    print('Total dials:', len(new_data), 'query dials:', query_dial)
    json.dump(new_data, open('data/seretod/processed_data_dev.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

def normalize():
    data=json.load(open('data/restructured_data_dev.json', 'r', encoding='utf-8'))
    count=0
    for dial in data:
        for turn in dial['content']:
            turn['用户意图']=turn['用户意图'].replace('（', '(').replace('）',')').replace('，',',')
            turn['客服意图']=turn['客服意图'].replace('（', '(').replace('）',')').replace('，',',')
            turn['用户意图']=turn['用户意图'].replace(',ent', ')(ent')
            turn['用户意图']=turn['用户意图'].replace('))', ')')
            turn['用户意图']=turn['用户意图'].replace('),(', ')(')
            turn['用户意图']=turn['用户意图'].replace(',(', ')(')
            turn['用户意图']=turn['用户意图'].replace('（', '(')
            turn['用户意图']=turn['用户意图'].replace('）', ')')
            turn['用户意图'] = turn['用户意图'].replace('通知','提供信息')
            turn['用户意图'] = turn['用户意图'].replace('求助-询问','求助-查询')
            turn['用户意图'] = turn['用户意图'].replace('求助-咨询','求助-查询')
            turn['用户意图'] = turn['用户意图'].replace('无效','其他')
            turn['用户意图'] = turn['用户意图'].replace('()','')
            if turn['用户意图'] == '':
                turn['用户意图'] = '其他'
            turn['客服意图']=re.sub(r'\(.*\)', '', turn['客服意图'])
            turn['客服意图']=re.sub(r'ent-[0-9]+-', '', turn['客服意图'])
            turn['客服意图'] = turn['客服意图'].replace('提供信息','通知')
            turn['客服意图'] = turn['客服意图'].replace('询问(流量包)','询问')
            turn['客服意图'] = turn['客服意图'].replace('询问(附加套餐)','询问')
            turn['客服意图'] = turn['客服意图'].replace('询问(主套餐)','询问')
            turn['客服意图'] = turn['客服意图'].replace('询问(业务)','询问')
            turn['客服意图'] = turn['客服意图'].replace('询问(用户状态)','询问')
            turn['客服意图'] = turn['客服意图'].replace('询问(业务费用)','询问')
            turn['客服意图'] = turn['客服意图'].replace('无效','其他')
            if turn['客服意图'] == '':
                turn['客服意图'] = '其他'
            info=re.findall(r'\((.*?)\)',turn['用户意图'])
            for e in info:
                if ',' in e:
                    e_list=[]
                    ent_id=None
                    for item in e.split(','):
                        if item.startswith('ent'):
                            ent_id=item[:5]
                            e_list.append(item)
                        elif item in ['业务','数据业务','套餐', '主套餐','附加套餐','国际漫游业务','流量包','长途业务','4G套餐','5G套餐']:
                            # type
                            e_list.append(item)
                        elif item in ['用户需求','用户要求','用户状态', '短信', '持有套餐','账户余额','流量余额', "话费余额", '欠费']:
                            # prop
                            e_list.append(item)
                        else:
                            # property of enity
                            if ent_id is not None:
                                e_list.append(ent_id+'-'+item)
                            else:
                                e_list.append(item)
                    ui=turn['用户意图']
                    turn['用户意图']=turn['用户意图'].replace(e, ')('.join(e_list))
                    print(ui, turn['用户意图'])
            '''
            if 'info' in turn:
                if 'ents' in turn['info']:
                    for ent in turn['info']['ents']:
                        if 'name' not in ent and 'ent-name' in ent:
                            ent['name']=ent.pop('ent-name')
                            count+=1
                        if 'id' not in ent and 'ent-id' in ent:
                            ent['id']=ent.pop('ent-id')
                            count+=1
                if 'triples' in turn['info']:
                    for tri in turn['info']['triples']:
                        if 'ent-name' not in tri and 'name' in tri:
                            tri['ent-name']=tri.pop('name')
                            count+=1
                        if 'ent-id' not in tri and 'id' in tri:
                            tri['ent-id']=tri.pop('id')
                            count+=1
            '''
    #print('修改实体三元组key个数:',count)
    json.dump(data, open('data/restructured_data_dev.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
 

def add_constraint():
    data=json.load(open('data/restructured_data_dev.json', 'r', encoding='utf-8'))
    query_num, query_wo_cons, query_cons_in_triple=0, 0, 0
    collected_turns=[]
    add_num=0
    for dial in data:
        for turn in dial['content']:
            collected=False
            ui=turn['用户意图']
            for query_intent in ['求助-查询', '询问', '主动确认']:
                if query_intent in ui:
                    query_num+=1
                    if query_intent+'(' not in ui:
                        added_cons=''
                        query_wo_cons+=1
                        if 'info' in turn:
                            for tri in turn['info']['triples']:
                                #tri['ent-name] in turn['用户'], if tri['ent-name']!='套餐'
                                # 上面这个约束可以不对主动确认生效，同时主动确认查询约束可以不加ent 
                                #询问类:tri['prop'] not in turn['用户']
                                #tri['prop']!='持有套餐''国内被叫''套餐外通话计费'
                                if ((tri['value'] in turn['用户']) and (query_intent == '主动确认') and (tri['prop'] not in ['持有套餐','国内被叫','套餐外通话计费'])) or ((tri['value'] in turn['客服']) and (query_intent in ['求助-查询', '询问']) and (tri['ent-name'] in turn['用户'] or tri['ent-name']=='套餐') and (tri['prop'] not in turn['用户'])):
                                    if 'ent-id' not in tri:
                                        continue
                                    if tri['ent-id'].startswith('ent'):
                                        cons=tri['ent-id']+'-'+tri['prop']
                                    elif tri['ent-id']=='NA' or (query_intent == '主动确认' and (tri['prop'] not in turn['用户'])):
                                        cons=tri['prop']
                                    if (cons not in added_cons) and (cons not in turn['用户意图']) and tri['prop']!='用户需求':
                                        added_cons+='('+cons+')'
                                    #query_cons_in_triple+=1
                                    if not collected:
                                        collected_turns.append(turn)
                                        collected=True
                        if added_cons!='':
                            turn['用户意图']=turn['用户意图'].replace(query_intent, query_intent+added_cons)
                            add_num+=1
                            break
    #print('查询意图:', query_num, '无约束的查询意图', query_wo_cons, '查询约束存在于三元组标注中', query_cons_in_triple)
    #print('询问意图:', inquiry_num, '无约束的询问意图', inquiry_wo_cons, '询问约束存在于三元组标注中', inquiry_cons_in_triple)
    #print('Collected turns:', len(collected_turns))
    #json.dump(collected_turns, open('data/temp.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('Add constraints num:', add_num)
    json.dump(data, open('data/restructured_data_dev.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

def process_unlabel():
    system_side = ['[SPEAKER 1]','[SPEAKER <A>]']
    user_side = ['[SPEAKER 2]','[SPEAKER <B>]']
    data=json.load(open('data/data_unlabel.json', 'r', encoding='utf-8'))#,object_pairs_hook=collections.OrderedDict
    dataset=[]
    for num in range(len(data)):
        dial = data[num]
        new_dial = []
        user = ''
        for turn in dial:
            new_turn ={}
            temp=list(turn.keys())
            if temp[0] not in system_side:
                if dial not in dataset:
                    dataset.append(dial)
                new_dial = []
                break
            else:
                if new_dial==[] and user == '': #first turn
                    if user_side[0] in turn:
                        user = turn[user_side[0]]  
                    elif user_side[1] in turn:
                        user = turn[user_side[1]]
                    else:
                        break
                else: 
                    new_turn['[SPEAKER 2]'] = user
                    if system_side[0] in turn:
                        new_turn['[SPEAKER 1]'] = turn[system_side[0]] 
                    elif system_side[1] in turn:
                        new_turn['[SPEAKER 1]'] = turn[system_side[1]]
                    else:
                        break
                    new_dial.append(new_turn)
                    if user_side[0] in turn:
                        user = turn[user_side[0]]  
                    elif user_side[1] in turn:
                        user = turn[user_side[1]]
                    else:
                        break
        if new_dial!=[]:
            dataset.append(new_dial)

            
    print(1)   
    #print('轮次——客服在前:', c1, '用户在前:', c2, '其他情况:', c3)
    #print('对话——客服在前:', len(dials1), '用户在前:', len(dials2), '其他情况:', len(dials3))
    json.dump(dataset, open('data/data_unlabel_processed.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    #json.dump(dials2, open('data/user_first.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    #json.dump(dials3, open('data/others.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
def check_unlabel():
    data=json.load(open('data/data_unlabel_processed.json', 'r', encoding='utf-8'))
    clean_dial=[]
    for dial in data:
        turn = dial[0]
        if '[SPEAKER 2]' not in turn :
            if '[SPEAKER <B>]' in turn:
                clean_dial.append(dial)
        elif '为您服务'  not in turn['[SPEAKER 2]'] :#or '您好' in turn['[SPEAKER 2]']:
            clean_dial.append(dial)
    print(1)
    json.dump(clean_dial, open('data/data_unlabel_process.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

def get_dialog_state():
    data=json.load(open('data/processed_data.json', 'r', encoding='utf-8'))
    for dial in data:
        kb = dial['KB']
        dial_id=dial['id']  
        user_state={} 
        hist = ''
        for turn in dial['content']:
            ui = turn['用户意图']
            user_state={}
            hist = hist + turn['用户']
            for ent,props in kb.items():
                for k,v in props.items():
                    # if k =='type' and v =='流量包'， v can change to '流量' 
                    value = v.split(',')[0]
                    if value in hist and k!='name':
                        if k in user_state:
                            if value not in user_state[k]:
                                user_state[k].append(value)
                        else:
                            user_state[k]=[value]
            intents = ui.split(',')
            for intent in intents:
                info=re.findall(r'\((.*?)\)', intent)
                k = intent.split('(')[0]
                if k in ['主动确认','询问','求助-查询']:
                    user_state[k] = []
                    for i in info:
                        tmp = i.split('-')[-1]
                        if tmp not in user_state[k]:
                            user_state[k].append(tmp)
            turn['user_state'] = user_state
    json.dump(data, open('data/seretod/dst_data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)        
            #for triple in turn['info']['triples']:
            #    if triple['value'] in turn['用户']:
            #        if  triple['ent-id'] == 'NA':
            # print(1)
            #hist = hist + turn['客服']
            
def get_global_KB():
    data=json.load(open('data/processed_data.json', 'r', encoding='utf-8'))
    ents = json.load(open('data/seretod/ents.json', 'r', encoding='utf-8'))
    ents_info = {}
    for type,ent in ents.items():
        ents_info[type] = {}
        for money in ent:
            ents_info[type][money] = {}
    user_info = {}
    sys_info = {}
    special_ent = []
    no_money_ent = []
    taocan = []
    ent_num = 0
    for dial in data:
        kb = dial['KB']
        for ent_name,ent in kb.items():
            if ent_name=='NA':
                ent_num = ent_num + 1
                for k,_ in ent.items():
                    if k not in user_info:
                        user_info[k] = 0
                    user_info[k] = user_info[k] + 1
                    if k == '持有套餐':
                        if ent not in taocan:
                            taocan.append(ent)
            else:
                if 'type' not in ent:
                    special_ent.append(ent)
                else:
                    if ent['type'] not in sys_info:
                        sys_info[ent['type']] = {}
                    if '业务费用' in ent:
                        money = ent['业务费用'].split(',')[0]
                        money = money.replace('钱','').replace('块','元').replace('的','')
                        if '元' not in money:
                            money = money + '元'
                        if ent['type'] in ents_info:
                            if money in ents_info[ent['type']]:
                                for k,v in ent.items():
                                    if k!='业务费用' and k!='type':
                                        if k not in ents_info[ent['type']][money]:
                                            ents_info[ent['type']][money][k] =[]
                                        if ','not in v:
                                            if v not in ents_info[ent['type']][money][k]:
                                                ents_info[ent['type']][money][k].append(v)
                                        else:
                                            value = v.split(',')
                                            for v1 in value:
                                                if v1 not in ents_info[ent['type']][money][k]:
                                                    ents_info[ent['type']][money][k].append(v1)
                                    
                        
                    else:
                        no_money_ent.append(ent)
    json.dump(ents_info, open('data/seretod/kb.json', 'w'), indent=2, ensure_ascii=False)
    print(1)



if __name__=='__main__':
    #get_dialog_state()
    #get_global_KB()
    #data_statistics()
    #restructure()
    #normalize()
    #add_constraint()
    #extract_local_KB()
    #process_unlabel()
    #check_unlabel()