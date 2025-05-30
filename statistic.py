import json
import numpy as np
import math
import scipy.stats as st
import argparse

from metrics import eval_end_to_end

def compute_P(N10, K):
    w=(abs(N10-K/2)-0.5)/math.sqrt(K/4)
    P=2*(1-st.norm(0,1).cdf(abs(w)))
    return P

def McNemar(list1, list2):
    c1=0
    c2=0
    for t1, t2 in zip(list1, list2):
        if t1 and not t2:
            c1+=1
        elif not t1 and t2:
            c2+=1
    P=compute_P(c1, c1+c2)
    return P

def matched_pair(list1, list2):
    n=len(list1)
    Z=[item1-item2 for item1, item2 in zip(list1, list2)]
    u=np.mean(Z)
    sigma=math.sqrt(sum([(z-u)**2 for z in Z])/(n-1))
    w=u*math.sqrt(n)/sigma
    P=2*(1-st.norm(0,1).cdf(abs(w)))
    return P

def test():
    #'experiments/gtdb_pseudo1/best_model/result.json'
    #'experiments/gtdb_pseudo2/best_model/result_0.1.json',
    #'experiments/gtdb_pseudo4/best_model/result_0.1.json',
    #'experiments/gtdb_pseudo_whole/best_model/result_0.1.json'

    result_path1 = ['experiments/jsa_1_scratch/best_model/result_0.1.json']

    #'experiments/jsa_1_scratch/best_model/result_0.1.json',
    #'experiments/jsa_2_scratch/best_model/result_0.1.json',
    #'experiments/jsa_4_scratch_gap/best_model/result_0.1.json',
    #'experiments/jsa_whole_scratch/last_model_1.650/result.json'

    # 'experiments/jsa_ebm_2/best_model/result.json'
    #'experiments/jsa_ebm_9_baseupon2/best_model/result.json'

    result_path2 = ['experiments/jsa_ebm_1/best_model/result.json']
    #'experiments/jsa_ebm_1/best_model/result.json'
    #'experiments/jsa_ebm_2/best_model/result.json'
    #'experiments/jsa_ebm_4_baseupon2/best_model/result.json'
    #'experiments/jsa_ebm_9_baseupon2/best_model/result.json'


    
    combined1 = []
    combined2 = []
    for p in result_path1:
        tmp = []
        results=json.load(open(p, 'r'))
        eval_result, test = eval_end_to_end(results, return_results=True)
        for t in test:
            score = 0
            score += t[0]['success']*100
            for turn in t:
                score = score + 2*turn['bleu']/len(t)
                #score1 = score + 2*turn['bleu']
                #tmp.append(score1)
            tmp.append(score)
        combined1.append(tmp)
    for q in result_path2:
        tmp1 = []
        results1=json.load(open(q, 'r'))
        eval_result1, test1 = eval_end_to_end(results1, return_results=True)
        for t in test1:
            score = 0
            score += t[0]['success']*100
            for turn in t:
                score = score + 2*turn['bleu']/len(t) #/len(t)
                #score1 = score + 2*turn['bleu']
                #tmp1.append(score1)
            tmp1.append(score)
        combined2.append(tmp1)
    print([matched_pair(combined2[i], combined1[i]) for i in range(1)])
    # or the pulled results can be compared
    # print(matched_pair([combined1[i]],combined2[i]) for i in range(4)])
    return

def get_analysis():


    return
    
if __name__=='__main__':
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("result_path1", type=str, help="The result file path (.json) of exp1")
    parser.add_argument("result_path2", type=str, help="The result file path (.json) of exp2")
    parser.add_argument("--method", type=str, default='mp', help="Matched pair test (mp) or McNemar test (mc)")
    args = parser.parse_args()
    list1 = json.load(open(args.result_path1, 'r'))
    list2 = json.load(open(args.result_path2, 'r'))
    test_func = McNemar if args.method=='mc' else matched_pair 
    print('p:', test_func(list1, list2))
    """
    #test()
    get_analysis()