
import re
schema={
    '业务':['业务','数据业务'],
    '数据业务':['数据业务'],
    '套餐':['套餐', '主套餐','4g套餐','5g套餐'],
    '主套餐':['主套餐', '4g套餐','5g套餐'],
    '附加套餐':['附加套餐', '国际漫游业务','流量包','长途业务'],
    '国际漫游业务':['国际漫游业务'],
    '流量包':['流量包'],
    '长途业务':['长途业务'],
    '4g套餐':['4g套餐'],
    '5g套餐':['5g套餐']
}
alter_props={
    '业务规则':['业务费用', '业务时长', '通话范围', '流量范围', '通话时长', '流量总量']
}
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

def query(KB, ent_id=None, ent_name=None, prop=None, ent_type=None, with_alter=True):
    if KB=={}:
        return None
    if ent_id is not None:
        if ent_id not in KB:
            return None
        value_string = None
        if KB[ent_id].get(prop, None):
            value_string = prop + ':'
            if ',' in KB[ent_id].get(prop, None):
                value_string = value_string + KB[ent_id].get(prop, None).split(',')[0]
            else:
                value_string = value_string + KB[ent_id].get(prop, None)
        return value_string
    elif ent_name is not None:# use entity name to query local KB
        value_string=None
        flag=0
        for en in ent_name.split(','):
            for key, ent in KB.items():
                if not key.startswith('ent'):
                    continue
                if en.lower() in ent['name'].split(','):
                    if with_alter:
                        if prop=='业务规则' and prop not in ent:
                            for alter_prop in alter_props[prop]:
                                if alter_prop in ent:
                                    if value_string:
                                        value_string = value_string + ',' + alter_prop +':' + (ent[alter_prop] if (',' not in ent[alter_prop])  else ent[alter_prop].split(',')[0])
                                    else:
                                        value_string = alter_prop +':' + (ent[alter_prop] if (',' not in ent[alter_prop])  else ent[alter_prop].split(',')[0])
                        elif prop in alter_props['业务规则'] and prop not in ent:
                            if ent.get('业务规则', None):
                                value_string = '业务规则' + ':' + ent.get('业务规则', None).split(',')[0]
                        else:
                            if ent.get(prop, None):
                                value_string = prop + ':'
                                if ',' in ent.get(prop, None):
                                    value_string = value_string + ent.get(prop, None).split(',')[0]
                                else:
                                    value_string = value_string + ent.get(prop, None)
                    else:
                        if ent.get(prop, None):
                            value_string = prop + ':'
                            if ',' in ent.get(prop, None):
                                value_string = value_string + ent.get(prop, None).split(',')[0]
                            else:
                                value_string = value_string + ent.get(prop, None)
                    if value_string is not None:# The corresponding value has been found from the current entity
                        flag=1
                        break
            if flag:
                break
        return value_string
    elif prop is not None:# query the user information, entity id or name is not needed
        user_info=['用户需求','用户要求','用户状态', '短信', '持有套餐','账户余额','流量余额', "话费余额", '欠费']
        value=None
        if 'NA' in KB:
            if KB['NA'].get(prop, None):
                value=prop +':' + KB['NA'][prop] if (',' not in KB['NA'][prop])  else KB['NA'][prop].split(',')[0]
            if value is None and prop in ['剩余话费', '话费余额']:
                for key in ['剩余话费', '话费余额']:
                    if key in KB['NA']:
                        value=key + ':' + (KB['NA'][key] if (',' not in KB['NA'][key])  else KB['NA'][key].split(',')[0])
                        break
        if value is None: # irregular query
            value_list=[]
            for key, ent in KB.items():
                if prop in ent:
                    if value:
                        if ent[prop] not in value:
                            value = value + ',' + (ent[prop] if (',' not in ent[prop])  else ent[prop].split(',')[0])
                    else:
                        value = ent[prop] if (',' not in ent[prop])  else ent[prop].split(',')[0]
        return value
    elif ent_type is not None: # Ask which entities are of this type， return list of ents
        names=[]
        for key, ent in KB.items():
            if not key.startswith('ent') or 'type' not in ent:
                continue
            if ent['type'].lower() in schema[ent_type.lower()]:
                name_string = ''
                for k,v in ent.items():
                    if k!='type' and k!='name' and (v.split(',')[0] not in name_string): # value not in name_string
                        if name_string:
                            name_string = name_string + ',' + k + ':' +  (v if (',' not in v)  else v.split(',')[0])
                        else:
                            name_string = k + ':' +  (v if (',' not in v)  else v.split(',')[0])
                names.append(name_string)
        return names if len(names)>0 else None
    else:
        return(serialize_kb(KB))
def query_old(KB, ent_id=None, ent_name=None, prop=None, ent_type=None, with_alter=True):
    if KB=={}:
        return None
    if ent_id is not None:
        if ent_id not in KB:
            return None
        return KB[ent_id].get(prop, None) if prop else None
    elif ent_name is not None:# use entity name to query local KB
        value=None
        flag=0
        for en in ent_name.split(','):
            for key, ent in KB.items():
                if not key.startswith('ent'):
                    continue
                if en.lower() in ent['name'].split(','):
                    if with_alter:
                        if prop=='业务规则' and prop not in ent:
                            value_list=[]
                            for alter_prop in alter_props[prop]:
                                if alter_prop in ent:
                                    value_list.append(ent[alter_prop])
                            value=','.join(value_list) if len(value_list)>0 else None
                        elif prop in alter_props['业务规则'] and prop not in ent:
                            value=ent.get('业务规则', None)
                        else:
                            value=ent.get(prop, None)
                    else:
                        value=ent.get(prop, None)
                    if value is not None:# The corresponding value has been found from the current entity
                        flag=1
                        break
            if flag:
                break
        return value
    elif prop is not None:# query the user information, entity id or name is not needed
        user_info=['用户需求','用户要求','用户状态', '短信', '持有套餐','账户余额','流量余额', "话费余额", '欠费']
        value=None
        if 'NA' in KB:
            value=KB['NA'].get(prop, None)
            if value is None and prop in ['剩余话费', '话费余额']:
                for key in ['剩余话费', '话费余额']:
                    if key in KB['NA']:
                        value=KB['NA'][key]
                        break
        if value is None: # irregular query
            value_list=[]
            for key, ent in KB.items():
                if prop in ent:
                    value_list.append(ent[prop])
            if value_list!=[]:
                value=','.join(value_list)
        return value
    elif ent_type is not None: # Ask which entities are of this type
        names=[]
        for key, ent in KB.items():
            if not key.startswith('ent') or 'type' not in ent:
                continue
            if ent['type'].lower() in schema[ent_type.lower()]:
                names.append(ent['name'])

        return names if len(names)>0 else None

def intent_query(KB, intent):
    KB_result=[]
    if '(' in UI:
        for intent in UI.split(','):
            if '('  not in intent:
                continue
            act=intent[:intent.index('(')]
            info=re.findall(r'\((.*?)\)', intent)
            #print(info)
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
    return KB_result

if __name__=='__main__':
    KB={
      "ent-1": {
        "name": "集团卡",
        "type": "主套餐",
        "业务费用": "十二块钱"
      },
      "ent-2": {
        "name": "安心包,十块钱一百兆的流量安心包,十块钱的流量安心包",
        "type": "流量包",
        "业务费用": "十块钱",
        "流量总量": "一百兆"
      },
      "ent-3": {
        "name": "套餐",
        "type": "主套餐"
      },
      "ent-4": {
        "name": "全球通的套餐",
        "type": "主套餐",
        "业务费用": "八元,二百三十八元"
      },
      "NA": {
        "用户需求": "把这个上网的关关闭一下"
      }
    }
    UI='询问(安心包-业务规则)'
    #{'ent-1': {'name': '三十八的,套餐', 'type': '主套餐', '流量总量': '一千兆', '业务费用': '三十八'}, 'ent-2': {'name': '流量包,包', 'type': '流量包', '业务规则': '十块钱一个G的二十块钱两个G'}, 'ent-3': {'name': '流量包,这个,十块钱一个g的,十块钱一个g的流量包', 'type': '流量包', '业务费用': '十块钱,十块', '流量总量': '一个G', '业务时长': '十二个月', '业务规则': '得到营业厅取消,流量还是不往下个月结转的'}, 'ent-4': {'name': '二十块钱两个g的', 'type': '流量包', '业务费用': '二十块钱', '流量总量': '两个G'}, 'NA': {'持有套餐': '三十八的'}}
    #sprint(UI)
    print(intent_query(KB, UI))

