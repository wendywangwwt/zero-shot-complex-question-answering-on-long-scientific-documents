import os
import pickle
import numpy as np
import pandas as pd

# def get_predictions(d_res_q):
#     d_res_pred = {}
#     d_agg_pred = {}
#     for model_name, annotations in d_res_q.items():
#         d_res_pred[model_name] = {}
#         d_agg_pred = {'id':ids_paper_eval,
#                          f'{model_name}_answers':[],
#                          f'{model_name}_scores':[]}
#         for id_paper_eval in ids_paper_eval:
#             d_res_pred[model_name][id_paper_eval] = annotations[id_paper_eval]
#             d_agg_pred[f'{model_name}_answers'].append([x['answer'] for x in annotations[id_paper_eval]])
#             d_agg_pred[f'{model_name}_scores'].append([np.round(x['score'],5) for x in annotations[id_paper_eval]])
#     return d_res_pred, d_agg_pred


def load_pred_all(dir_pk):
    fns = [fn for fn in os.listdir(dir_pk) if fn.endswith('.pk') and '|' in fn]
    d_res = {}
    for fn in fns:
        q_id, model_name = fn[:-3].split('|')
        q_id = int(q_id.split('_')[-1][1:]) # e.g., d_res_q6 -> 6
        model_name = model_name.replace('__','/') # e.g., ahotrod__albert_xxlargev1_squad2_512 -> ahotrod/albert_xxlargev1_squad2_512
        if not q_id in d_res:
            d_res[q_id] = {}
        d_res[q_id][model_name] = pickle.load(open(os.path.join(dir_pk,fn),'rb'))
    return d_res

def load_pred(q='q1',dir_pk='.'):
    fns = [x for x in os.listdir(dir_pk) if x.startswith(f'd_res_{q}') and x.endswith('.pk')]
    # print(fns)
    assert len(fns) > 0, f'cannot find any file'
    d_res_q = {}
    for fn in fns:
        q_id, model_name = fn[:-3].split('|')
        model_name = model_name.replace('__','/')
        d_res_q[model_name] = pickle.load(open(os.path.join(dir_pk,fn),'rb'))
    
    model_names = list(d_res_q.keys())
    ids_paper = list(d_res_q[model_names[0]].keys())
    return d_res_q, model_names, ids_paper




def merge_answers_by_location(a1,a2,l1,l2):
    """
    a1: answer 1 in string
    a2: answer 2 in string
    l1: a tuple to indicate location of the answer (start index, end index) for answer 1
    l2: a tuple to indicate location of the answer (start index, end index) for answer 2
    
    Return
        merged answer in string and updated location tuple
    """
    l3 = (min(l1[0],l2[0]),max(l1[1],l2[1]))
    offset = l3[0]
    l1_offset = (l1[0]-offset,l1[1]-offset)
    l2_offset = (l2[0]-offset,l2[1]-offset)

    res = ''
    if l1[0]<l2[0]:
        res += a1
        res += a2[(l1_offset[1]-l2_offset[0]):]
    else:
        res += a2
        res += a1[(l2_offset[1]-l1_offset[0]):]
    
    return res, l3


def merge_answers_by_text(answers,scores,locations):
    answers_recorded = []
    scores_recorded = []
    locations_recorded = []
    for answer, score, location in zip(answers, scores, locations):
        flag_to_add = True
        answer, location = refine_answer(answer, location)
        for answer_recorded, score_recorded, location_recorded in zip(answers_recorded, scores_recorded, locations_recorded):
            if answer in answer_recorded:
                flag_to_add = False
                break
            elif answer_recorded in answer:
                # replace the short answer with the long answer while keeping its position the same
                answers_recorded = [answer if x == answer_recorded else x for x in answers_recorded]
                # for the score we still use the higher score recorded earlier
                locations_recorded = [location if x == location_recorded else x for x in locations_recorded]
                flag_to_add = False # because we have already added the answer
                break
            else:
                pass
        if flag_to_add:
            answers_recorded.append(answer)
            scores_recorded.append(score)
            locations_recorded.append(location)
    return answers_recorded, scores_recorded, locations_recorded
    

def refine_answer(a,l,c_to_ignore=[' ',',','.','-','(',';']):
    """
    Remove unwanted trailing or leading characters
    """
    idx_s = None
    idx_e = None
    
    # remove trailing ")" if "(" is not found in the string
    if a.endswith(')') and '(' not in a:
        a = a[:-1]
        l = (l[0],l[1]-1)
    
    for i,c in enumerate(list(a)):
        if c in c_to_ignore and idx_s is None:
            pass
        else:
            idx_s = i
            break
    
    for i,c in enumerate(list(a[::-1])):
        if c in c_to_ignore and idx_e is None:
            pass
        else:
            idx_e_reversed = i
            idx_e = len(a)-idx_e_reversed # convert to the index used by the original string order
            break
    #print(idx_s,idx_e_reversed, idx_e)
    try:
        return a[idx_s:idx_e], (l[0]+idx_s,l[1]-idx_e_reversed)
    except:
        print(a,l)

def merge_answers_(answers,scores,locations,threshold=0,return_location=False,sort=False):
    answers_recorded = []
    scores_recorded = []
    locations_recorded = []
    for answer,score,location in zip(answers,scores,locations):
        if len(answers_recorded) == 0 or score > threshold:
            # merge answers if the new answer has overlap with one of the recorded answers
            if answer not in answers_recorded:
                flag_no_overlap = True
                for i, (answer_recorded,location_recorded) in enumerate(zip(answers_recorded,locations_recorded)):                              
                    if location[0] > location_recorded[1] or location[1] < location_recorded[0]:
                        pass
                    else:
                        flag_no_overlap = False
                        # update answer and location; for score, we will keep the highest which is the one that gets added to the list earlier
                        answer, location = merge_answers_by_location(answer, answer_recorded, location, location_recorded)
                        break # for now we don't consider situation where one new answer can be merged with two non-overlapped recorded answers (e.g., 'def' to ['abcd','fg'])
                        
                if not flag_no_overlap:
                    answers_recorded.remove(answer_recorded)
                    locations_recorded.remove(location_recorded)                            
                
                if not return_location: # if location indices are needed, white spaces will be kept so the reference location is correct
                    answer = answer.strip()
                answers_recorded.append(answer)
                locations_recorded.append(location)

                if flag_no_overlap:
                    scores_recorded.append(score)
            else: # ignore answers that have exactly the same text as an already recorded one
                pass

    # merge answers where one is completely overlapped by another
    answers_recorded, scores_recorded, locations_recorded = merge_answers_by_text(answers_recorded, scores_recorded, locations_recorded)

    if sort and len(answers_recorded) > 0:
        # sort answers by the start index of location
        combined = list(zip(answers_recorded, scores_recorded, locations_recorded))
        combined = sorted(combined,key=lambda x: x[2][0]) # sort by the first element of the third element (location tuple)
        answers_recorded, scores_recorded, locations_recorded = map(list, zip(*combined))
        
    return answers_recorded, scores_recorded, locations_recorded


def qa_answer_separate_list(l, topk_cutoff=1000):
    """
    d: a list of qa answers where each entry is a dict containing 4 keys (answer, score, start, end)
    this separates one list of top answers into 3 lists: answers, scores, locations
    """
    # print('-'*50)
    # print(l)
    answers = [x['answer'] for x in l][:topk_cutoff] # cannot strip the string here because it would mess up the results from merge_answers_by_location()
    scores = [np.round(x['score'],5) for x in l][:topk_cutoff]
    locations = [(x['start'],x['end']) for x in l][:topk_cutoff]
    return answers, scores, locations

def get_predictions(d_res_q, ids_include=[],return_location=False, return_score=False, score_cutoff=0, topk_cutoff=1000, merge_answers=True):
    """
    Parse and return formatted predictions from dictionary.
    Answer format example:
        {'<id_paper>':[{'answer':'<>','score':<>,'start':<>,'end':<>},{...}]}

    d_res_q: a dictionary of responses from model prediction
    ids_include: a list of document ids to be included in the output
    return_location: whether to return location of answer (i.e., start and end index)
    return_score: whether to return confidence score
    score_cutoff: a threshold below which the answers will be dropped
        if the value is a number, it works as a typical threshold
        if the value is a dictionary, it should be in the form of {"model1":0.2, "model2":0.3} 
        where the key refers to the model name in d_res_q and the value refers to the score
        cutoff for this particular model
    topk_cutoff: only get the top k predictions
    """
    assert isinstance(score_cutoff,int) or isinstance(score_cutoff,float) or isinstance(score_cutoff,dict), "score_cutoff should be an integer, a float, or a dictionary"
    if isinstance(score_cutoff,dict):
        for k in score_cutoff.keys():
            assert k in d_res_q, f"key {k} in score_cutoff cannot be found in d_res_q"

    d_res_pred = {}
    if len(ids_include) == 0:
        ids_include = list(list(d_res_q.values())[0].keys())
    d_agg_pred = {'id':ids_include}
        
    for model_name, annotations in d_res_q.items():
        d_res_pred[model_name] = {}
        d_agg_pred[f'{model_name}_answers'] = []
        threshold = score_cutoff[model_name] if isinstance(score_cutoff,dict) else score_cutoff
        
        if return_score:
            d_agg_pred[f'{model_name}_scores'] = []
        if return_location:
            d_agg_pred[f'{model_name}_locations'] = []
            
        for id_paper_eval in ids_include:
            d_res_pred[model_name][id_paper_eval] = annotations[id_paper_eval][:topk_cutoff]
            answers, scores, locations = qa_answer_separate_list(annotations[id_paper_eval], topk_cutoff)

            # remove answers that has no letter
            l_idx_drop = []
            for i, answer in enumerate(answers):
                if sum([x.isalpha() for x in answer]) == 0:
                    l_idx_drop.append(i)
            remove_element_by_index(answers,l_idx_drop)
            remove_element_by_index(scores,l_idx_drop)
            remove_element_by_index(locations,l_idx_drop)

            if merge_answers:
                answers_recorded, scores_recorded, locations_recorded = merge_answers_(answers,scores,locations,threshold,return_location)
            else:
                answers_recorded = answers
                scores_recorded = scores
                locations_recorded = locations
            d_agg_pred[f'{model_name}_answers'].append(answers_recorded)
            if return_score:
                d_agg_pred[f'{model_name}_scores'].append(scores_recorded)
            if return_location:
                d_agg_pred[f'{model_name}_locations'].append(locations_recorded)
                  
    return d_res_pred, d_agg_pred


def get_predictions_msh(d_res_q, ids_include=[],return_location=False, return_score=False, score_cutoff=0.01, topk_cutoff=1000, merge_answers=True,merge_all_answers=False):
    """
    Parse and return formatted predictions from dictionary.
    Answers to Q2 is different - for each generated ML/NLP category identified in a given article, Q2 collects the top5 extracted answers towards
    the particular category.
    Answer format example:
        {'<id_paper>':[{'<category>':[{'answer':'<>','score':<>,'start':<>,'end':<>},{...}],
                        '<category>':[{'answer':'<>','score':<>,'start':<>,'end':<>},{...}]},
                        ...]}

    d_res_q: a dictionary of responses from model prediction
    ids_include: a list of document ids to be included in the output
    return_location: whether to return location of answer (i.e., start and end index)
    return_score: whether to return confidence score
    score_cutoff: a threshold below which the answers will be dropped
        if the value is a number, it works as a typical threshold
        if the value is a dictionary, it should be in the form of {"model1":0.2, "model2":0.3} 
        where the key refers to the model name in d_res_q and the value refers to the score
        cutoff for this particular model
    topk_cutoff: only get the top k predictions
    """
    assert isinstance(score_cutoff,int) or isinstance(score_cutoff,float) or isinstance(score_cutoff,dict), "score_cutoff should be an integer, a float, or a dictionary"
    if isinstance(score_cutoff,dict):
        for k in score_cutoff.keys():
            assert k in d_res_q, f"key {k} in score_cutoff cannot be found in d_res_q"

    if merge_all_answers and not merge_answers:
        merge_answers = True
    
    d_res_pred = {}
    if len(ids_include) == 0:
        ids_include = list(list(d_res_q.values())[0].keys())
    d_agg_pred = {'id':ids_include}
        
    for model_name, annotations in d_res_q.items():
        d_res_pred[model_name] = {}
        d_agg_pred[f'{model_name}_answers'] = []
        threshold = score_cutoff[model_name] if isinstance(score_cutoff,dict) else score_cutoff
        
        if return_score:
            d_agg_pred[f'{model_name}_scores'] = []
        if return_location:
            d_agg_pred[f'{model_name}_locations'] = []
            
        for id_paper_eval in ids_include:
            # Not implemented: in d_res_pred the answers are NOT truncated by topk - this at the moment is not used so not implemented
            anno =  annotations[id_paper_eval] 
            for d_answer in anno:
                for k, output in d_answer.items():
                    try:
                        d_answer[k] = output[:topk_cutoff]
                    except:
                        print(model_name, id_paper_eval, d_answer)
            d_res_pred[model_name][id_paper_eval] = anno
            if merge_all_answers:
                l_output = [d for d_answer in anno for output in d_answer.values() for d in output[:topk_cutoff]]
                answers, scores, locations = qa_answer_separate_list(l_output, topk_cutoff=10000)
                
                # remove answers that has no letter
                l_idx_drop = []
                for i, answer in enumerate(answers):
                    if sum([x.isalpha() for x in answer]) == 0:
                        l_idx_drop.append(i)
                remove_element_by_index(answers,l_idx_drop)
                remove_element_by_index(scores,l_idx_drop)
                remove_element_by_index(locations,l_idx_drop)

                answers_recorded, scores_recorded, locations_recorded = merge_answers_(answers,scores,locations,threshold,return_location,sort=True)
                    
                d_agg_pred[f'{model_name}_answers'].append(answers_recorded)
                if return_score:
                    d_agg_pred[f'{model_name}_scores'].append(scores_recorded)
                if return_location:
                    d_agg_pred[f'{model_name}_locations'].append(locations_recorded)
            else:
                l_answers = []
                l_scores = []
                l_locations = []
                for d_answer in annotations[id_paper_eval]:                        
                    for cat,l_output in d_answer.items():
                        answers, scores, locations = qa_answer_separate_list(l_output, topk_cutoff)
            
                        # remove answers that has no letter
                        l_idx_drop = []
                        for i, answer in enumerate(answers):
                            if sum([x.isalpha() for x in answer]) == 0:
                                l_idx_drop.append(i)
                        remove_element_by_index(answers,l_idx_drop)
                        remove_element_by_index(scores,l_idx_drop)
                        remove_element_by_index(locations,l_idx_drop)
            
                        if merge_answers:
                            answers_recorded, scores_recorded, locations_recorded = merge_answers_(answers,scores,locations,threshold,return_location)
                        else:
                            answers_recorded = answers
                            scores_recorded = scores
                            locations_recorded = locations
                        l_answers.append(answers_recorded)
                        l_scores.append(scores_recorded)
                        l_locations.append(locations_recorded)
                    
                d_agg_pred[f'{model_name}_answers'].append(l_answers)
                if return_score:
                    d_agg_pred[f'{model_name}_scores'].append(l_scores)
                if return_location:
                    d_agg_pred[f'{model_name}_locations'].append(l_locations)
                  
    return d_res_pred, d_agg_pred


def summary_list_cols(df,colnames_list=['answer_q1'],agg_func='np.mean'):
    d_res = {'colname':colnames_list,
             'avg_count':[],
               'avg_len':[],
               'avg_empty':[]}
    for c in colnames_list:   
        l_a = df[c].values.tolist()
        count = 0
        count_neg = 0
        length = 0
        
        for sublist_a in l_a:
            count += len(sublist_a)
            #print(np.mean([len(a.strip().split()) for a in sublist_a]))
            length += eval(agg_func)([len(a.strip().split()) for a in sublist_a]) if len(sublist_a) > 0 else 0
            if len(sublist_a) == 0:
                count_neg += 1
        
        count /= (df.shape[0] - count_neg) # when calculating average count and length, only divde by number of valid entries
        length /= (df.shape[0] - count_neg)
        count_neg /= df.shape[0]
        
        d_res['avg_count'].append(count)
        d_res['avg_len'].append(length)
        d_res['avg_empty'].append(count_neg)
    return pd.DataFrame(d_res)

def summary_text_cols(df,colnames_text=['text_cleaned']):
    d_res = {'colname': colnames_text,
               'avg_len':[],
               'std_len':[]
               }
    for c in colnames_text:   
        l_text = df[c].values.tolist()
        l_len = [len(text.strip().split()) for text in l_text]
        length = np.mean(l_len)
        std = np.std(l_len)
        
        d_res['avg_len'].append(length)
        d_res['std_len'].append(std)
    return pd.DataFrame(d_res)




# def get_predictions_simple(d_res_q, include_score=False):
#     """
#     Convert the predictions into the following output format:
#     |       |answer_1|answer_2|...|
#     -------------------------------
#     |model_1|value_1 |value_2 |...|
#     ...

#     include_score: whether to attach the confidence score in answer value
#     """
#     model_names = list(d_res_q.keys())
#     d_res_pred = {'model':model_names}
    
#     num_answers = len(d_res_q[model_names[0]][0]) # assume all entries has the same number of answers (usually this should hold)
#     d_res_pred = {**d_res_pred, **{f'answer_{i}':[] for i in range(1,num_answers+1)}}

#     for model_name in model_names:
#         for i in range(1,num_answers+1):
#             a = d_res_q[model_name][0][i-1]['answer'] # index is i-1
#             if include_score:
#                 a += f" ({round(d_res_q[model_name][0][i-1]['score'],3)})"
#             d_res_pred[f'answer_{i}'].append(a)
#     return pd.DataFrame(d_res_pred)

def get_predictions_simple(d_res_q, include_score=False):
    """
    Convert the predictions into the following output format:
    |       |answer_1|answer_2|...|
    -------------------------------
    |model_1|value_1 |value_2 |...|
    ...

    include_score: whether to attach the confidence score in answer value
    """
    model_names = list(d_res_q.keys())
    l_idx = list(d_res_q[model_names[0]].keys())
    
    d_res_pred = {'model':[model_name for idx in l_idx for model_name in model_names ],
                  'id':[idx for idx in l_idx for model_name in model_names ]}
    
    
    num_answers = len(d_res_q[model_names[0]][l_idx[0]]) # assume all entries has the same number of answers (usually this should hold)
    d_res_pred = {**d_res_pred, **{f'answer_{i}':[] for i in range(1,num_answers+1)}}

    for idx in l_idx:
        for model_name in model_names:
            for i in range(1,num_answers+1):
                a = d_res_q[model_name][idx][i-1]['answer'] # index is i-1
                if include_score:
                    a += f" ({round(d_res_q[model_name][idx][i-1]['score'],5)})"
                d_res_pred[f'answer_{i}'].append(a)
    return pd.DataFrame(d_res_pred)


def remove_element_by_index(l_value,l_index):
    if len(l_index) > 0:
        for index in sorted(l_index, reverse=True):
            del l_value[index]



def combine_answer(l1,l2, empty_list_action='both'):
    """
    empty_list_action: if one list is empty and the other is not, 
                       "empty" returns an empty list
                       "both" returns the non-empty one (equivalent to combining them)
    """
    if len(l1) * len(l2) == 0:
        if empty_list_action == 'empty':
            return []
    return list(set(l1+l2))


def msh_to_df(d_pred, ids_include=[], models_include=['deepset/deberta-v3-large-squad2'], keys_included=[],merge_answers=True):
    d_res = {'model_name':[],
             'id':[],
             'answer_key':[],
             'answer_val':[],
             'answer_score':[]}
    if len(ids_include) == 0:
        model = list(d_pred.keys())[0]
        ids_include = list(d_pred[model].keys())
    for model_name in models_include:
        for id_entry in ids_include:
            l_pred = d_pred[model_name][id_entry]
            for d_answer in l_pred:
                for answer_key, l_answers in d_answer.items():
                    if len(keys_included) > 0 and answer_key.lower() not in keys_included:
                        continue # pass this one
                    else:
                        
                        l_answer_val = []
                        l_answer_score = []
                        
                        for d_answer_topk in l_answers:
                            answer_val = d_answer_topk['answer']
                            answer_score = d_answer_topk['score']
                            flag_pass = False
                            if len(l_answer_val) > 0 and merge_answers:
                                for i,answer_val_collected in enumerate(l_answer_val): # compare against each collected answer value 
                                    if answer_val in answer_val_collected:
                                        flag_pass = True
                                        break
                                    elif answer_val_collected in answer_val:
                                        l_answer_val[i] = answer_val # replace the collected answer with the current longer answer
                                        flag_pass = True
                                        break
    
                                if not flag_pass:
                                    l_answer_val.append(answer_val)
                                    l_answer_score.append(answer_score)     
                            else:
                                l_answer_val.append(answer_val)
                                l_answer_score.append(answer_score)
                            
                        d_res['model_name'] += [model_name] * len(l_answer_val)
                        d_res['id'] += [id_entry] * len(l_answer_val)
                        d_res['answer_key'] += [answer_key] * len(l_answer_val)
                        d_res['answer_val'] += l_answer_val
                        d_res['answer_score'] += l_answer_score
    return pd.DataFrame(d_res)