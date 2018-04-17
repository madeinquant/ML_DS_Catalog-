# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 23:27:56 2018

@author: santanu
Apriori Algorithm with Multiple Minimum Support 
"""

import numpy as np
from collections import defaultdict
import pandas as pd
import itertools
import time

def process_level_1(path,global_min_supp,beta,item_contraint,items_cannot,items_must):
    f = open(path,'r')
    level_1_support = defaultdict(int)
    level_1_min_support = defaultdict(float)
    level_1_min_support_final = defaultdict(float)
    level_1_items = set()
    trans_count = 0
    rec_ = ""
    rec_ = f.readline()
    while rec_ <> '':
        items = rec_.split('\n')[0]
        if items <> '':
            items = items.split(" ")
        items = [int(item) for item in items]
        
        for item in items:
            level_1_support[item] +=1 
        trans_count +=1
        rec_ = f.readline()
        
      
        
    for item,count_ in zip(level_1_support.keys(),level_1_support.values()):
        supp = count_/float(trans_count)
        level_1_support[item] = supp
        if beta*supp > global_min_supp:
            level_1_min_support[item] = beta*supp
        else:
            level_1_min_support[item] = global_min_supp
        if supp > level_1_min_support[item]:
            level_1_items.add(item)
    f.close()
    level_1_items = list(level_1_items)
      
    for item in level_1_items:
        level_1_min_support_final[item] = level_1_min_support[item]
    keys = np.array(level_1_min_support_final.keys())
    values = np.array(level_1_min_support_final.values())
    indices = np.argsort(values)
    keys = keys[indices]
    level_1_items = keys 
    if item_contraint == 'Y':
        level_1_items = item_constraint(level_1_items,items_cannot,items_must,level_1_min_support)
    
    return level_1_support,level_1_min_support,level_1_items,trans_count

def level_2_candidate_gen(level_1_support,level_1_min_support,level_1_items,phi_thresh):
    level_2_potential_candidates = set()
    level_2_potential_candidates_min_supp = defaultdict(float)
    
    for i in xrange(len(level_1_items)):
        for j in xrange(i+1,len(level_1_items)):
            if (np.absolute(level_1_min_support[level_1_items[i]] - level_1_min_support[level_1_items[j]]) < phi_thresh):
                level_2_potential_candidates.add((level_1_items[i],level_1_items[j]) ) 
                level_2_potential_candidates_min_supp[(level_1_items[i],level_1_items[j])] = min([level_1_min_support[level_1_items[i]],level_1_min_support[level_1_items[j]]])
                
                   
    return list(level_2_potential_candidates),level_2_potential_candidates_min_supp
    
def level_2_prune(level_2_potential_candidates,level_2_potential_candidates_min_supp,level_1_min_support,path,item_contraint,items_cannot,items_must):
    f = open(path,'r')
    level_2_support = defaultdict(int)
    level_2_items = []
    trans_count = 0
    rec_ = ""
    rec_ = f.readline()
    while rec_ <> '':
        items = rec_.split('\n')[0]
        if items <> '':
            items = items.split(" ")
        items = np.array([int(item) for item in items])
        items_min_supp = np.array([level_1_min_support[item] for item in items])
        indices =  np.argsort(items_min_supp)         
        items = list(items[indices])
        
        for i in xrange(len(items)):
            for j in xrange(i+1,len(items)):
                level_2_support[(items[i],items[j])] +=1
                
        trans_count +=1
        
        rec_ = f.readline()
        
    for p_item in level_2_potential_candidates:
        supp = level_2_support[p_item]/float(trans_count)
        level_2_support[p_item] = supp
        if supp > level_2_potential_candidates_min_supp[p_item]:
            level_2_items.append(p_item)
            
    if item_contraint == 'Y':
        level_2_items = item_constraint(list(level_2_items),items_cannot,items_must,level_1_min_support)
        
        
    return list(level_2_items),level_2_support
 
def level_n_candidate_generation(level_n_minus_1_items,level_1_min_support,phi_thresh):
    level_n_minus_pd = pd.DataFrame(np.array(level_n_minus_1_items))
    cols = list(level_n_minus_pd.columns)
    merge_pd = level_n_minus_pd.merge(level_n_minus_pd,on=cols[:-1])
    merge_pd_cols = list(merge_pd.columns)
    final_col_name = str(cols[-1])
    final_col_name_x = final_col_name + str('_x')
    final_col_name_y = final_col_name + str('_y')
    
    merge_pd['order_diff'] = merge_pd[final_col_name_x].apply(lambda x:level_1_min_support[x]) - merge_pd[final_col_name_y].apply(lambda x:level_1_min_support[x])
    merge_pd['potential_select'] = merge_pd['order_diff'].apply(lambda x:1 if x < 0 else 0)
    merge_pd['order_diff_phi_criterion'] = merge_pd['order_diff'].apply(lambda x:1 if -1*x < phi_thresh else 0 )
    merge_pd = merge_pd[merge_pd['potential_select'] == 1]
    merge_pd = merge_pd[merge_pd['order_diff_phi_criterion'] == 1]
    candidates_potential_bp  =  list(merge_pd[merge_pd_cols].values)
    candidates_potential_bp = [tuple(k) for k in candidates_potential_bp]
    candidates_potential_ap = set(candidates_potential_bp)
    
    for item in candidates_potential_bp:
        item_list = list(item)
        item_len = len(item_list)
        delete_flag = 'N'
        i = 0
        while ((i < item_len) & (delete_flag == 'N')):
            s = item_list[0:i] + item_list[i+1:]
            s = tuple(s)
            if (item_list[1] in s) or (level_1_min_support[item_list[1]] == level_1_min_support[item_list[2]]):
                if s not in list(level_n_minus_1_items):
                    candidates_potential_ap.remove(item)
                    delete_flag = 'Y'
                    break
            i+=1 
    candidates_potential_ap = list(candidates_potential_ap)
    candidates_potential_ap = [ tuple(k) for k in candidates_potential_ap]
    return candidates_potential_ap
    
   
def level_n_prune(level_n_potential_candidates,level_1_min_support,path,level,item_contraint,items_cannot,items_must):
    
    f = open(path,'r')
    level_n_support = defaultdict(int)
    level_n_items = []
    trans_count = 0
    rec_ = ""
    rec_ = f.readline()
    while rec_ <> '':
        items = rec_.split('\n')[0]
        if items <> '':
            items = items.split(" ")
        items = np.array([int(item) for item in items])
        items_min_supp = np.array([level_1_min_support[item] for item in items])
        indices =  np.argsort(items_min_supp)         
        items = list(items[indices])
        
        for k in itertools.combinations(items,level):
            level_n_support[k] +=1 
            
                
        trans_count +=1
        
        rec_ = f.readline()
        
    for p_item in level_n_potential_candidates:
        supp = level_n_support[p_item]/float(trans_count)
        level_n_support[p_item] = supp
        if supp > level_1_min_support[p_item[0]]:
            level_n_items.append(p_item)
    if item_contraint == 'Y':
        level_n_items = item_constraint(list(level_n_items),items_cannot,items_must,level_1_min_support)
    
    return list(level_n_items),level_n_support  

def item_constraint(in_list,items_cannot,items_must,level_1_min_support):
    out_list_1 = []
    out_list_2 = []
    out_list_3 = set()
    item_cannot_sort = []
    full_items_cannot =  []
    
    for item in items_cannot:
        item_support = [level_1_min_support[k] for k in item]
        indices = np.argsort(np.array(item_support))
        item_ = list(np.array(item)[indices])
        item_cannot_sort.append(item_)
        
    items_support = [level_1_min_support[k[0]] for k in item_cannot_sort]
    indices_1 = np.argsort(np.array(items_support))
    items_cannot_sort_order = []
    for item_ind in indices_1:
        items_cannot_sort_order.append(item_cannot_sort[item_ind])
    for i in xrange(len(item_cannot_sort)) :
        for j in xrange(i+1,len(item_cannot_sort)):
            full_items_cannot.append(set(item_cannot_sort[i] + item_cannot_sort[j]))
            
        
        
    
    for c in in_list:
        if type(c) == tuple:
            elem = set(c)
        else:
            elem = set([c])
        if elem not in full_items_cannot:
            out_list_1.append(c)
        if len(elem.intersection(set(items_must))) > 0 :
            out_list_2.append(c)
    out_list_3 =   set(out_list_1).intersection(set(out_list_2))
    out_list_3 = list(out_list_3)
    return out_list_3

def process_para(path,global_min_supp,beta,phi_thresh,items_cannot,items_must,item_contraint):
    level_1_support,level_1_min_support,level_1_items,trans_count =  process_level_1(path,global_min_supp,beta,item_contraint,items_cannot,items_must)
    level_2_potential_candidates,level_2_potential_candidates_min_supp = level_2_candidate_gen(level_1_support,level_1_min_support,level_1_items,phi_thresh)
    level_2_items,level_2_support = level_2_prune(level_2_potential_candidates,level_2_potential_candidates_min_supp,level_1_min_support,path,item_contraint,items_cannot,items_must)
    items_full_list = list(level_1_items) + list(level_2_items)
    items_full_supp = {}
    items_full_supp.update(level_1_support)
    items_full_supp.update(level_2_support)
    level_n_items = level_2_items
    level = 3
    
    while len(level_n_items) <> 0:
        level_n_potential_candidates = level_n_candidate_generation(level_n_items,level_1_min_support,phi_thresh)
        len(level_n_potential_candidates)
        level_n_items,level_n_support = level_n_prune(level_n_potential_candidates,level_1_min_support,path,level,item_contraint,items_cannot,items_must)
        items_full_list = items_full_list + list(level_n_items)
        items_full_supp.update(level_n_support)
        level +=1 
    results_dict =  {}
    results = pd.DataFrame()
    for c in items_full_list:
        results_dict[c] = items_full_supp[c]
    results['Item Sets'] = results_dict.keys()
    results['Support Probability'] = results_dict.values()
    results['Support Count'] = results['Support Probability'].apply(lambda x:int(x*trans_count))
    results = results.sort_values(['Support Probability'],ascending=False)
    return results
    
    
    
        
        
if __name__ == '__main__':
     
    start_time = time.time()
    path = 'C:\\Users\\santanu\\Desktop\\Data Mining\\dataset (1)\\dataset\\retail1\\retail1.txt'
    global_min_supp = 0.05        
    beta = 0.5
    phi_thresh = 0.05
    items_cannot = [[1534,1582],[1394,1989],[225,1215],[1816,1834],[1534,1943]]
    items_must = [1534,1394,225,1816]
    item_contraint = 'Y'
    results = process_para(path,global_min_supp,beta,phi_thresh,items_cannot,items_must,item_contraint)
    end_time = time.time()
    print results
    print "Processing Time:",end_time - start_time,"secs"
    
    
    
    
    
