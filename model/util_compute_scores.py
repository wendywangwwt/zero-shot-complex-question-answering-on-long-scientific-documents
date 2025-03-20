import numpy as np
import pandas as pd
import os
import copy
from evaluate_squadv2 import compute_f1, normalize_answer

from util_load_data import get_predictions
from util_hf import pool

import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import torch
import gc

def compute_metrics_matches(df,colname_gt='answer_q1',colnames_pred=[],score_cutoff={'matches_similar':0.7,'f1':0.7},device='cpu',cache_dir='./cache/huggingface/models',similar_match_model='intfloat/e5-mistral-7b-instruct'):
    """
    Add calculated results (matches) to the input dataframe.
    3 columns are produced for each metric:
    <metric>|<model>: list, the matched answer from eval set that can find a match in the pred set
    <metric>_pred|<model>: list, the matched answer from pred set that corresponds to those listed in <metric>|<model>
    <metric>_ratio|<model>: float, the ratio of the eval set that can find a match in the pred set, len(<metric>|<model>) / len(eval_set)

    score_cutoff: score cutoff used by matches_similar where the score is a cosine similarity score between text embeddings
    """
    l_gt = df[colname_gt].values.tolist() # used for returning matched answers
    l_gt_normalized = [apply_string_process_to_list(sublist,method='normalize_answer') for sublist in l_gt] # used for calculating scores

    # sentenceTransformer cannot load models with half precision :/
    model_name = similar_match_model
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir, device_map = device)
    model = AutoModel.from_pretrained(model_name,cache_dir=cache_dir, torch_dtype = torch.bfloat16, device_map = device)
    print(f'model {model_name} loaded')

    for c in colnames_pred:
        l_pred = df[c].values.tolist()
        l_pred_normalized = [apply_string_process_to_list(sublist,method='normalize_answer') for sublist in l_pred]

        d_metrics = get_matches(l_gt,l_gt_normalized,l_pred,l_pred_normalized,model=model,tokenizer=tokenizer,score_cutoff=score_cutoff,device=device,cache_dir=cache_dir)
        
        for metric_name,d_metric in d_metrics.items():
            for metric_type, metric_value in d_metric.items():
                df[f'{metric_name}_{metric_type}|'+c] = metric_value
        print(f'Finished column {c}')

def get_matches(l_gt,l_gt_normalized,l_pred,l_pred_normalized,model="intfloat/e5-mistral-7b-instruct",tokenizer=None,score_cutoff=0.8,device='cpu',cache_dir='./cache/huggingface/models'):
    """
    l_gt: a list of lists where each sublist is the ground truth answers for 1 document
    l_gt_normalized: a list of lists where each sublist is the ground truth answers for 1 document; each string of answer has been cleaned with function normalize_answer
    l_pred: a list of lists where each sublist is the predicted answers for 1 document
    l_pred_normalized: a list of lists where each sublist is the predicted answers for 1 document; each string of answer has been cleaned with function normalize_answer
    model: model used to compute embeddings
        if the value is a string, load the model
        if the value is a model, use it directly
    score_cutoff: a threshold below which the answers will be dropped
        if the value is a number, it works as a typical threshold
        if the value is a dictionary, it should be in the form of {"model1":0.2, "model2":0.3} 
        where the key refers to the metric name (currently available for ['matches_similar', 'f1']) and the value refers to the score
        cutoff for this particular metric
    """  
    d_res = {'matches_exact':{'gt':[],'pred':[],'ratio':[]},
             'matches_similar':{'gt':[],'pred':[],'ratio':[]},
             'mentions':{'gt':[],'pred':[],'ratio':[]},
             'mentions_advanced':{'gt':[],'pred':[],'ratio':[]},
             'f1':{'gt':[],'pred':[],'max':[]}, # max here is the average of max f1 over all gold answers
             'precision':{'gt':[],'pred':[],'max':[]},
             'recall':{'gt':[],'pred':[],'max':[]},
             'f1_normalized':{'gt':[],'pred':[],'max':[]}, # average of max normalized by total length of predicted answers; f1/precision/recall only takes into account the BEST predicted answer, so in ablation study of parameter topk all of they _tend_ to monotonically increase as topk increases (not always) because there are more predicitons and the merged predictions are longer; hence here we normalize these scores by averaging f1/precision/recall by the number of words in all predicted answers given a document
             'precision_normalized':{'gt':[],'pred':[],'max':[]},
             'recall_normalized':{'gt':[],'pred':[],'max':[]},
             # same as Qasper baseline, for multiple answer spans, concatenate all spans and then use SQuAD's evaluator
             'f1_concat':{'gt':[],'pred':[],'max':[]},
             'precision_concat':{'gt':[],'pred':[],'max':[]},
             'recall_concat':{'gt':[],'pred':[],'max':[]} 
             }
    
    assert isinstance(score_cutoff,int) or isinstance(score_cutoff,float) or isinstance(score_cutoff,dict), "score_cutoff should be an integer, a float, or a dictionary"
    if isinstance(score_cutoff,dict):
        for k in score_cutoff.keys():
            assert k in d_res, f"key {k} in score_cutoff cannot be found in available metrics {list(d_res.keys())}"
    else:
        score_cutoff = {'matches_similar':score_cutoff,'f1':score_cutoff}

    if isinstance(model,str):
        model = SentenceTransformer("intfloat/e5-mistral-7b-instruct",cache_folder=cache_dir,device=device)
        model.max_seq_length = 4096

    # for each document
    for sublist_gt, sublist_gt_normalized, sublist_pred, sublist_pred_normalized in zip(l_gt,l_gt_normalized,l_pred,l_pred_normalized):
        d_res_sublist = {'matches_exact':{'gt':[],'pred':[],'ratio':[]},
             'matches_similar':{'gt':[],'pred':[],'ratio':[]},
             'mentions':{'gt':[],'pred':[],'ratio':[]},
             'mentions_advanced':{'gt':[],'pred':[],'ratio':[]},
             'f1':{'gt':[],'pred':[],'max':[]}, # max here is the average of max f1/precision/recall over all gold answers
             'precision':{'gt':[],'pred':[],'max':[]},
             'recall':{'gt':[],'pred':[],'max':[]},
             'f1_normalized':{'gt':[],'pred':[],'max':[]}, # average of max normalized by total length of predicted answers
             'precision_normalized':{'gt':[],'pred':[],'max':[]},
             'recall_normalized':{'gt':[],'pred':[],'max':[]},
             'f1_concat':{'gt':[],'pred':[],'max':[]},
             'precision_concat':{'gt':[],'pred':[],'max':[]},
             'recall_concat':{'gt':[],'pred':[],'max':[]} 
             }

        for a_gt, a_gt_normalized in zip(sublist_gt, sublist_gt_normalized):
            for a_pred, a_pred_normalized in zip(sublist_pred, sublist_pred_normalized):
                # matches exact
                if a_gt_normalized == a_pred_normalized:
                    d_res_sublist['matches_exact']['gt'].append(a_gt)
                    d_res_sublist['matches_exact']['pred'].append(a_pred)
                # mentions
                if a_gt in a_pred:
                    d_res_sublist['mentions']['gt'].append(a_gt)
                    d_res_sublist['mentions']['pred'].append(a_pred)
                # mentions advanced: special processing for abbreviations, punctuation
                # use original strings here and run normalize_answer(drop_punc=False/True) in compute_mentions_advanced()
                if compute_mentions_advanced(a_gt,a_pred):
                    d_res_sublist['mentions_advanced']['gt'].append(a_gt)
                    d_res_sublist['mentions_advanced']['pred'].append(a_pred)

        # keep unique values
        for metric_name, d_metric in d_res_sublist.items():
            for metric_type, metric_value in d_metric.items():
                if isinstance(metric_value,list):
                    if len(metric_value) > 0:
                        d_res_sublist[metric_name][metric_type] = list(set(metric_value))
        
        # embedding-based similarity matching: note that original strings are used to avoid potential issues caused by text cleaning
        d_res_sublist['matches_similar']['gt'], d_res_sublist['matches_similar']['pred'] = compute_embeddings(sublist_gt, sublist_pred, model=model,tokenizer=tokenizer,device=device,score_cutoff=score_cutoff['matches_similar'],cache_dir=cache_dir)

        # f1 score calculation based on squad v2 evaluation script: look at overlapping words
        d_res_f1_pr = compute_f1_(sublist_gt, sublist_gt_normalized, sublist_pred, sublist_pred_normalized, score_cutoff=score_cutoff['f1'])
        for metric_name, d_metric in d_res_sublist.items():
            for metric_type, metric_value in d_metric.items():
                if metric_name in ['f1','precision','recall','f1_normalized','precision_normalized','recall_normalized']:
                    d_res_sublist[metric_name][metric_type] = d_res_f1_pr[metric_name][metric_type]

         # f1 score calculation based on Qasper's approach: concate answer spans then look at overlapping words
        d_res_f1_pr_concat = compute_f1_(sublist_gt, sublist_gt_normalized, sublist_pred, sublist_pred_normalized, score_cutoff=score_cutoff['f1'],concat=True)
        for metric_name, d_metric in d_res_sublist.items():
            for metric_type, metric_value in d_metric.items():
                if metric_name in ['f1_concat','precision_concat','recall_concat']:
                    d_res_sublist[metric_name][metric_type] = d_res_f1_pr_concat[metric_name.split('_')[0]][metric_type]

        # TO-DO: word match score based on tf-idf: e.g., ('RoBERTa','neural network model RoBERTa') is a better match than ('neural network model','neural network model RoBERTa')
        

        # calculate ratios
        if len(sublist_gt) > 0:
            d_res_sublist['matches_exact']['ratio'] = len(d_res_sublist['matches_exact']['gt']) / len(sublist_gt)
            d_res_sublist['matches_similar']['ratio'] = len(d_res_sublist['matches_similar']['gt']) / len(sublist_gt)
            d_res_sublist['mentions']['ratio'] = len(d_res_sublist['mentions']['gt']) / len(sublist_gt)
            d_res_sublist['mentions_advanced']['ratio'] = len(d_res_sublist['mentions_advanced']['gt']) / len(sublist_gt)          
            
        else:
            d_res_sublist['matches_exact']['ratio'] = 1 if len(sublist_pred) == 0 else 0
            d_res_sublist['matches_similar']['ratio'] = 1 if len(sublist_pred) == 0 else 0
            d_res_sublist['mentions']['ratio'] = 1 if len(sublist_pred) == 0 else 0
            d_res_sublist['mentions_advanced']['ratio'] = 1 if len(sublist_pred) == 0 else 0
        
        # update final results
        for metric,d_metric in d_res.items():
            for answer_type in d_metric.keys():
                d_res[metric][answer_type].append(d_res_sublist[metric][answer_type])
    
    return d_res
            
## Matches and Mentions
def compute_mentions_advanced(s1,s2,verbose=0):
    """
    1. For those answers with abbreviations, count also pred answers that only mention abbreviations / full names
        e.g., for gt "support vector machines (SVMs)", a pred answer "SVMs" or "support vector machines" should both be counted as a valid one
    2. If no match, remove punctuation & concate characters, then try again (in pdf extraction sometimes hyphens/white-space are removed, e.g., cross-validation -> crossvalidation)
    s1: ground truth answer
    s2: pred answer
    """

    # special handling for R (this shows up as a gold answer of software)
    if s1 == 'R':
        s2 = normalize_answer(s2)
        if 'r' in s2.split():
            return True
        else:
            return False

    # condition 1: we need to keep punctuation because here we rely on "()" to identify abbreviations
    s1 = normalize_answer(s1, drop_punc=False)
    s2 = normalize_answer(s2, drop_punc=False)
    
    if verbose > 0:
        print(s1,s2)
    if s1.endswith(')') and '(' in s1:
        if verbose > 0:
            print('s1 has parentheses')
        l_s1 = s1[:-1].split('(') # remove ")" and then split by "("
        l_s1 = [s.strip() for s in l_s1]
        for s in l_s1:
            if s in s2:
                return True
    else:
        if verbose > 0:
            print('s1 has no parentheses')
        if s1 in s2:
            return True

    if s1.endswith(']') and '[' in s1:
        if verbose > 0:
            print('s1 has brackets')
        l_s1 = s1[:-1].split('[') # remove "]" and then split by "["
        l_s1 = [s.strip() for s in l_s1]
        for s in l_s1:
            if s in s2:
                return True
    else:
        if verbose > 0:
            print('s1 has no parentheses')
        if s1 in s2:
            return True

    # condition 2: ignore punctuation, concate characters, and match again
    s1 = normalize_answer(s1).replace(' ','')
    s2 = normalize_answer(s2).replace(' ','')
    if s1 in s2:
        return True
    return False
        
def apply_string_compute_to_list(l1,l2,method='compute_mentions_advanced',return2=False):
    """
    Apply a method for string comparison to 2 lists of strings.
    return2: return the matched strings in l2 rather than l1
    """
    res = []
    for a1 in l1:
        for a2 in l2:
            if eval(method)(a1,a2):
                if return2:
                    res.append(a2)
                else:
                    res.append(a1)
    return list(set(res))

def apply_string_process_to_list(l,method='normalize_answer'):
    """
    Apply a method for string processing to 1 list of strings.
    """
    return [eval(method)(s) for s in l]


def compute_embeddings(l1,l2,model="intfloat/e5-mistral-7b-instruct",tokenizer=None,cache_dir='./cache/huggingface/models',score_cutoff=0.85,device='cpu',
                       chunk_size=10):
    """
    Compute embedding-based similarity score between 2 lists of strings, then return the true and pred answers that
    have a score above the cutoff.
    In order to run embeddings on A30 (24G), the data is passed to the model chunk by chunk.
    
    model: model used to compute embeddings
        if the value is a string, load the model
        if the value is a model, use it directly
    chunk_size: used for l2 (list of predicted answers), run embeddings for l2 chunk by chunk to avoid oom
    """
    if len(l1) > 0 and len(l2):
        if isinstance(model,str):
            tokenizer = AutoTokenizer.from_pretrained(model,cache_dir=cache_dir, device_map = 'cuda:0')
            model = AutoModel.from_pretrained(model,cache_dir=cache_dir, torch_dtype = torch.bfloat16, device_map = 'cuda:0')
            
        # score computation is adapted from https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/mteb_except_retrieval_eval.py
        instruct = 'Instruct: Retrieve semantically similar text.\nQuery: ' # instruction used in intfloat/e5-mistral-7b-instruct for STS task; modify instruction according to model configuration if other models are used
        max_length = 4096
        
        input_texts = [instruct + s for s in l1] + [instruct + s for s in l2]
        batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')

        if batch_dict['input_ids'].shape[0] > 15:
            
            # too much memory cost; run embeddings separately for l1 and l2
            # l1 is usually fine, not larger than 15
            input_texts = [instruct + s for s in l1]
            batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = model(**batch_dict)
            embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'],'last') # last
            embeds_l1 = F.normalize(embeds, p=2, dim=-1)

            scores = np.empty((embeds_l1.shape[0],0)) # create an empty np array to collect scores by chunk

            # l2 could have problem in Q2 where there might be many collected predictions
            input_texts = [instruct + s for s in l2]
            while len(input_texts) > 0:
                input_texts_chunk = input_texts[:chunk_size]
                batch_dict = tokenizer(input_texts_chunk, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)
            
                outputs = model(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'],'last') # last
                embeds_l2 = F.normalize(embeds, p=2, dim=-1)

                scores_chunk = (embeds_l1 @ embeds_l2.T).cpu().detach().float().numpy()
                
                scores = np.hstack((scores,scores_chunk)) # concate the scores horizontally
                if len(input_texts) > chunk_size:
                    input_texts = input_texts[chunk_size:]
                else:
                    input_texts = []

        else:
            batch_dict = batch_dict.to(device)
            outputs = model(**batch_dict)
            embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'],'last') # last
            embeds = F.normalize(embeds, p=2, dim=-1)
            scores = (embeds[:len(l1)] @ embeds[len(l1):].T).cpu().detach().float().numpy() # float() to convert bfloat16 to 32 for numpy to handle

        l_matches_similar_gt = []
        l_matches_similar_pred = []
        for i,s1 in enumerate(l1):
            if np.max(scores[i]) > score_cutoff:
                    l_matches_similar_pred += list(np.array(l2)[np.where(scores[i]>score_cutoff)]) # retrieve all pred answers that meet the criterion
                    l_matches_similar_gt.append(s1)
        return l_matches_similar_gt, list(set(l_matches_similar_pred))
    else:
        return [],[]

## Precision, Recall, F1
def compute_f1_(sublist_gt, sublist_gt_normalized, sublist_pred, sublist_pred_normalized, score_cutoff=0.8, concat=False):
    """
    Compute F1 score which relies on the overlapping words.

    concat: True to concatenate all answer strings in each sublist before computation, so that multiple answer spans get merged
            into one big string to count overlapping words, which is used by Qasper baseline: https://github.com/allenai/qasper-led-baseline/blob/afd0fb96bf78ce8cd8157639c6f6a6995e4f9089/scripts/evaluator.py#L75
            False to treat each answer span separately
    """
    d_res = {'f1':{'gt':[],'pred':[],'max':[]}, 
             'precision':{'gt':[],'pred':[],'max':[]},
             'recall':{'gt':[],'pred':[],'max':[]},
             'f1_normalized':{'gt':[],'pred':[],'max':[]}, 
             'precision_normalized':{'gt':[],'pred':[],'max':[]},
             'recall_normalized':{'gt':[],'pred':[],'max':[]},
            }

    if concat:
        sublist_gt = [', '.join(sublist_gt)]
        sublist_gt_normalized = [', '.join(sublist_gt_normalized)]
        sublist_pred = [', '.join(sublist_pred)]
        sublist_pred_normalized = [', '.join(sublist_pred_normalized)]

    # original strings are used here instead of normalized ones because compute_f1() starts with calling using normalize_answer()
    for a_gt in sublist_gt:
        max_f1 = 0
        max_precision = 0
        max_recall = 0
        count_matched_f1 = 0
        count_matched_recall = 0
        count_matched_recall = 0
        for a_pred in sublist_pred:
            # print(compute_f1(a_gt, a_pred))
            f1,precision,recall = compute_f1(a_gt, a_pred)
            
            if f1 > max_f1:
                max_f1 = f1
            if f1 > score_cutoff:
                d_res['f1']['pred'].append(a_pred) # one prediction could be added for multiple times as it gets compared against different gold answers
                count_matched_f1 += 1
                if count_matched_f1 == 1:
                    d_res['f1']['gt'].append(a_gt)

            if precision > max_precision:
                max_precision = precision
            if precision > score_cutoff:
                d_res['precision']['pred'].append(a_pred) # one prediction could be added for multiple times as it gets compared against different gold answers
                count_matched_recall += 1
                if count_matched_recall == 1:
                    d_res['precision']['gt'].append(a_gt)

            if recall > max_recall:
                max_recall = recall
            if recall > score_cutoff:
                d_res['recall']['pred'].append(a_pred) # one prediction could be added for multiple times as it gets compared against different gold answers
                count_matched_recall += 1
                if count_matched_recall == 1:
                    d_res['recall']['gt'].append(a_gt)
                    
        d_res['f1']['max'].append(max_f1)
        d_res['precision']['max'].append(max_precision)
        d_res['recall']['max'].append(max_recall)
    # for squad v2, one question has one answer that could be written in multiple ways
    # so there is a set of gold answers for each question for a given document
    # the overall f1 is the max f1 over all gold answers
    # here however, we have a set of gold answers where each encodes unique information
    # and ideally all of them should be recalled
    # the overall f1 is the average of max f1 over all gold answers
    d_res['f1']['max'] = np.mean(d_res['f1']['max'])
    d_res['precision']['max'] = np.mean(d_res['precision']['max'])
    d_res['recall']['max'] = np.mean(d_res['recall']['max'])
    
    d_res['f1']['pred'] = list(set(d_res['f1']['pred']))
    d_res['precision']['pred'] = list(set(d_res['precision']['pred']))
    d_res['recall']['pred'] = list(set(d_res['recall']['pred']))

    # normalize f1/precision/recall by (len_pred / len_gt)
    len_gt = sum([len(a_gt.split()) for a_gt in sublist_gt_normalized])
    len_pred = sum([len(a_pred.split()) for a_pred in sublist_pred_normalized])
    ratio_pred_to_gt = (len_pred + 0.00001) / (len_gt+ + 0.00001)  # shift to avoid len_pred or len_gt being 0
    d_res['f1_normalized']['max'] = d_res['f1']['max'] / ratio_pred_to_gt
    d_res['precision_normalized']['max'] = d_res['precision']['max'] / ratio_pred_to_gt
    d_res['recall_normalized']['max'] = d_res['recall']['max'] / ratio_pred_to_gt
    
    return d_res
        

def summary(df, return_value='mean', pivot=False, calculate_harmonic_mean=True):
    """
    Summarize document-level scores to model level.
    df: a dataframe created by function "compute_metrics_matches" that contains metrics on document level
    pivot: if true, return a wide table with metric as column and model as row; else return a long table
    """
    
    df_metrics = pd.DataFrame(df.drop(columns=['id']).describe().loc[return_value]).reset_index().rename(columns={'mean':'score'})
    df_metrics['model'] = df_metrics['index'].apply(lambda x: x.split('|')[1])
    df_metrics['metric'] = df_metrics['index'].apply(lambda x: x.split('|')[0])
    df_metrics = df_metrics.drop(columns=['index'])

    if pivot:
        df_metrics = df_metrics.pivot(index='model',columns='metric')
        # after pivoting, both column and row have multi-index, so we flatten them before returning the dataframe
        df_metrics.columns = df_metrics.columns.droplevel(0)
        #df_metrics.index = df_metrics.index.droplevel(0)
        if return_value == 'mean' and calculate_harmonic_mean:
            df_metrics['harmonic_mean'] = df_metrics.apply(lambda x: 2/(1/x['mentions_advanced_ratio']+1/x['matches_similar_ratio']),axis=1)
            df_metrics = df_metrics.sort_values('harmonic_mean',ascending=False)
        return df_metrics
    else:
        return df_metrics


def summary_topk(d_res_q, df_true, df_others=None, colname_gt='answer_1', colnames_pred=['model_answer_1'],topk_max=20,topk_min=5,
                 dir_save_intermediate=None,
                 device='cpu',cache_dir='./cache/huggingface/models',
                 similar_match_model='intfloat/e5-mistral-7b-instruct'):
    """
    Given a range of topk, calculate the metrics for each value between the range (both min and max included).
    
    d_res_q: a list of prediction results for a given question yielded by function load_pred, for example:
        d_res_q, model_names, ids_paper = load_pred('q1',dir_pk=dir_pred)
    df_true: a pandas dataframe of two columns (id, <colname_gt>) that contains the document id and the ground truth answer
    colname_gt: column name of the ground truth answer
    colnames_pred: a list of column names for the predictions, one model output per column
    topk_max: the max topk to calculate the results
    topk_min: the min topk to calculate the results
    dir_save_intermediate: directory to save intermediate file (i.e., document-level metrics at each topk value); do not save anything if value is None
    """
    l_topk = list(range(topk_min,topk_max+1))
    
    df_topk_summary_mean = pd.DataFrame()
    df_topk_summary_std = pd.DataFrame()
    if dir_save_intermediate:
        df_res = pd.DataFrame()
    for topk in l_topk:
        d_res_q_pred, d_agg_q_pred = get_predictions(d_res_q, df_true['id'].values.tolist(), topk_cutoff=topk)
    
        df_q = df_true[['id',colname_gt]].merge(pd.DataFrame(d_agg_q_pred),how='left',on='id')
        if df_others is not None:
            df_q = df_q.merge(df_others,how='left',on='id')
        compute_metrics_matches(df_q,colname_gt=colname_gt,colnames_pred=colnames_pred,device=device,cache_dir=cache_dir,similar_match_model=similar_match_model)

        if dir_save_intermediate:
            df_res = pd.concat([df_res, df_q], ignore_index=True)
    
        df_summary_mean = summary(df_q,'mean')
        df_summary_mean['topk'] = topk
        df_summary_std = summary(df_q,'std')
        df_summary_std['topk'] = topk
        
        df_topk_summary_mean = pd.concat([df_topk_summary_mean, df_summary_mean], ignore_index=True)
        df_topk_summary_std = pd.concat([df_topk_summary_std, df_summary_std], ignore_index=True)

    if dir_save_intermediate:
        path_fn = os.path.join(dir_save_intermediate,f'df_topk_{colname_gt}_{topk_min}_{topk_max}.csv')
        df_res.to_csv(path_fn,index=False)
        print('Document-level metrics calculated for each topk value has been saved to',path_fn)
    return df_topk_summary_mean, df_topk_summary_std


def get_categories(x, pipeline_tg, 
                   prompt, system_content=None):
    """
    x: an example sentence used in the prompt
    pipeline_tg: a huggingface text generation pipeline
    """
    if x.strip() == '':
        return []
    
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": prompt+x 
                    })
    
    prompt = pipeline_tg.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )
    
    terminators = [
        pipeline_tg.tokenizer.eos_token_id,
        pipeline_tg.tokenizer.convert_tokens_to_ids("<|eot_id|>") # llama 3 needs this
    ]
    
    outputs = pipeline_tg(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.00000001,
        top_p=1,
    )
    res = outputs[0]["generated_text"][len(prompt):]
    try:
        return eval(res) # the raw output hopefully will be ["a","b"...]
    except:
        try: 
            # mostly the type of formatting issue observed in our data with the given prompt & input 
            # is additional double quotes added to the output categories
            assert res.startswith('[') and res.endswith(']'), f"output ({res}) does not start with or end with brackets"
            res = res[1:-1] # remove brackets
            l_cat = [cat.strip().strip('"') for cat in res.split(',')] # remove all leading & trailing double quotes
            return l_cat
        except:
            try:
                # example:
                # Logistic Regression (LR), the five different machine learning classification techniques, downstream natural language processing (NLP), the SVM, RFC and LRtechniques
                # output: [""Machine Learning" techniques: Logistic Regression, Support Vector Machine, Random Forest", ""Natural Language Processing""]"
                
                print(x)
                print(res)
                res = res.strip('"')
                assert res.startswith('[') and res.endswith(']'), f"output ({res}) does not start with or end with brackets"
                res = res[1:-1] # remove brackets
                l_cat = [cat.strip().strip('"') for cat in res.split(',')]
                for i,cat in enumerate(l_cat):
                    if ':' in cat:
                        cat = cat.split(':')[-1].strip().strip('"')
                        l_cat[i] = cat
                print(l_cat)
                print('FIXED')
                return l_cat
            except:
                print(x)
                print(res)
                print('-'*50)
                return res # return the original text if cannot be evaluated


def compute_metrics_between_lists(l_gt,l_matched,l_pred=None):
    """
    This function computes metrics based on three lists: the list of sublists of ground truth answer, list of sublists of matched answers,
    and list of sublists of predicted answers.
    l_pred is used to calculate scores when l_gt has empty sublist (in this case, sublist_gt is empty, sublist matched is empty, but we need
    to know whether sublist_pred is also empty to determine the score). If all sublists in l_gt are non-empty, then l_pred is not needed.
    """
    l_match_ratio = []
    
    for i, (sublist_gt, sublist_matched),  in enumerate(zip(l_gt, l_matched)):
        if isinstance(sublist_gt,str):
            sublist_gt = eval(sublist_gt)

        if isinstance(sublist_matched,str):
            sublist_gt = eval(sublist_matched)
        
        if len(sublist_gt) == 0:
            assert l_pred is not None, f'l_gt contains empty sublist, you must pass l_pred to determine the score'
            sublist_pred = l_pred[i]
            if isinstance(sublist_pred,str):
                sublist_pred = eval(sublist_pred)
            sublist_pred = [x for x in sublist_pred if len(x.strip())>0]
            score = 1 if len(sublist_pred) == 0 else 0
            print(sublist_gt,sublist_pred,score)
        else:
            score = len(sublist_matched) / len(sublist_gt)
        l_match_ratio.append(score)
    print(l_match_ratio)
    return np.mean(l_match_ratio)