import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path_text','-pt',default='df_text.pk',help='path to the text file, currently assumes to be a pickle of pandas df or a csv or an excel sheet')
parser.add_argument('--col_text','-ct',default='text',help='column name for the text field')
parser.add_argument('--path_questions','-pq',default='questions.json',help='a json list of question prompts; see README.md for how to configure this file')
parser.add_argument('--dir_output',default='pred',help='directory to save QA results')
parser.add_argument('--device',default=0,type=int,help='gpu id to run huggingface pipeline')
parser.add_argument('--hf_token',default='',type=str,help='huggingface token, needed to access restricted models such as llama')
parser.add_argument('--hf_cache_dir',default='./cache/huggingface/models',help='cache dir for huggingface models')
parser.add_argument('--rag_model',default="meta-llama/Meta-Llama-3-8B-Instruct",help="huggingface model name used for RAG; it needs to be compatible with huggingface's text-generation pipeline")

args = parser.parse_args()
path_text = args.path_text
col_text = args.col_text
path_questions = args.path_questions
dir_output = args.dir_output
device = args.device
token = args.hf_token
cache_dir = args.hf_cache_dir
rag_model = args.rag_model

TOKEN_PLACEHOLDER = '__PLACEHOLDER__'

# these settings should be configured before importing hf
os.environ['HF_DATASETS_CACHE']= cache_dir
os.environ['HF_HOME'] = cache_dir

import transformers
from transformers import pipeline
from datasets import Dataset
import torch
import gc
import time
import pickle
import json
import itertools
from pprint import pprint
import pandas as pd
import numpy as np

from util_load_data import *
from util_compute_scores import *
print('transformers version:',transformers.__version__)
print('pandas version:',pd.__version__)
print('numpy version:',np.__version__)

def main():
    os.makedirs(dir_output,exist_ok=True)
    
    #-------- Step 1: run extractive QA model on text --------
    print('* Running extractive QA model on text...')
    fn_log = 'inference.log' # this file logs the completed inference
    t_s_total = time.time()

    # remove questions that require multi-single-hop in this single-hop qa process
    qs = {k:v for k,v in d_qs.items() if len(v[2]) == 0 or v[2][0] != 'msh'}
    print(f'{len(qs)} questions without multi-single-hop setting to be processed')
    pprint(qs)

    # l_combinations_to_infer = [(qname,k) for qname in qs.keys() for k in model_checkpoints]
    l_combinations_to_infer = []
    for qname,qsetting in qs.items():
        for model_checkpoint in qsetting[1]:
            l_combinations_to_infer.append((qname,model_checkpoint))
    print(f'{len(l_combinations_to_infer)} model inference runs')
    try:
        with open(os.path.join(dir_output,fn_log)) as f:
            log = [l.split('\t') for l in f.readlines()]
        for qname,k,_ in log:
            l_combinations_to_infer.remove((qname,k))
        print(f'{len(log)} files are found in the log, removing from the config for inference...')
    except Exception as e:
        print('skipped loading from log file:',e)
        pass
    print(f'{len(l_combinations_to_infer)} final model inference runs')
    pprint(l_combinations_to_infer)

    for qname,qsetting in qs.items():
        q = qsetting[0]
        qmodels = qsetting[1]
        print('*'*20,f'q{qname}',q,'*'*20)
        d_res = {}
        t_s_q = time.time()
        for k in qmodels: #k,v in d_models.items():
            if (qname,k) in l_combinations_to_infer:
                t_s_q_k = time.time()
                pipeline_qa = pipeline(task="question-answering", 
                                       model=k,
                                       device=device)
                d_res[k] = {}
                print(f'inference with model {k}...')
                
                for i in df_text.index:
                    context = df_text[col_text].loc[i]
                    id_paper = df_text['id'].loc[i]
                    # print('q',q)
                    # print('context',context)
                    d_res[k][id_paper] = pipeline_qa(question=q, context=context, top_k=30, #device=0,
                                                     batch_size=128,doc_stride=128)
                    if i % 100 == 0 or i+1 == df_text.shape[0]:
                        print(f'Done {i+1}/{df_text.shape[0]}')
                        print(len(d_res),len(d_res[k]))

                time_used = time.time() - t_s_q_k
                print(f'Finished model {k} on q{qname}, elapsed time', time_used)
                pickle.dump(d_res[k],open(os.path.join(dir_output,f'd_res_q{qname}|{k.replace("/","__")}.pk'),'wb'))
    
                # update log file
                with open(os.path.join(dir_output,fn_log),'a') as f:
                    f.write('\t'.join([str(qname),k,str(time_used)])+'\n')
            else:
                print(f'Skipped model {k} on q{qname}: found record in log file')
                
        print(f'Finished all models on q{qname}, elapsed time', time.time() - t_s_q)
        
    print(f'Finished all models on all questions, elapsed time', time.time() - t_s_total)
    print()

    # remove pipeline_tg and release resource
    if 'pipeline_qa' in locals():
        del pipeline_qa
        gc.collect()
        torch.cuda.empty_cache()
    
    #-------- Step 2 RAG-Enhanced Multi-Span Entity Extraction --------
    print('* Running RAG-enhanced multi-span entity extraction...')
    qs = {k:v[2][1:] for k,v in d_qs.items() if len(v[2]) > 0 and v[2][0] == 'rag'}
    print(f'{len(qs)} questions with rag setting to be processed')
    pprint(qs)

    model_id = rag_model
    print(f'Use model {model_id} with text-generation pipeline')
    pipeline_tg = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device,
    )
    
    for qname,qparams in qs.items():
        print(f'Processing question {qname} with params {qparams}...')
        if os.path.exists(dir_output+f'df_q{qname}_category.pk'):
            print(f'Skipped question q{qname}: found output')
        else:
            qprompt = qparams[0]
            qmerge = bool(qparams[1].replace('merge',''))
            d_res_q, model_names, ids_paper = load_pred(f'q{qname}',dir_pk=dir_output)
            d_res_q_pred, d_agg_q_pred = get_predictions(d_res_q, [],topk_cutoff=20,merge_answers=qmerge)
            
            df_q = pd.DataFrame(d_agg_q_pred)#[['id']+[x+'_answers' for x in models]]
            overwrite = True
            count = 0
            for c in df_q.columns:
                #if c in models:
                if c.endswith('_answers'):
                    if c+'_categories' in df_q.columns:
                        if not overwrite:
                            print('overwrite False, skip',c)
                            continue
                    df_q[c] = df_q[c].apply(lambda x: list(set([e.strip() for e in x])))
                    df_q[c+'_categories'] = df_q[c].apply(lambda x: get_categories(', '.join(x), pipeline_tg, prompt=qprompt))
                    print('Finished column', c)
                    count += 1

            # only save id & categories
            pickle.dump(df_q[['id']+[c for c in df_q.columns if c.endswith('_categories')]],open(os.path.join(dir_output,f'df_q{qname}_category.pk'),'wb'))
    
    # remove pipeline_tg and release resource
    del pipeline_tg
    gc.collect()
    torch.cuda.empty_cache()

    print('Finished')
    print()

    #-------- Step 3 Multi-Single-Hop --------
    print('* Running multi-single-hop...')
    qs = {k:v for k,v in d_qs.items() if len(v[2]) > 0 and v[2][0] == 'msh'}
    print(f'{len(qs)} questions with msh setting to be processed')
    pprint(qs)

    for qname,qsetting in qs.items():
        q = qsetting[0]
        qmodels = qsetting[1]
        qparams = qsetting[2]
        print(f'Processing question {qname} with setting {qsetting}...')

        qname_entity = qparams[1]
        if len(d_qs[qname_entity][2]) > 0 and d_qs[qname_entity][2][0] == 'rag':
            print(f'Bridge entities are loaded from q{qname_entity} RAG-enhanced categorized answers...')
            df_q = pickle.load(open(os.path.join(dir_output,f'df_q{qname_entity}_category.pk'),'rb'))
            cols_combine = [c for c in df_q.columns if c.endswith('_answers_categories')]
            
        elif len(d_qs[qname_entity[2]]) == 0:
            print(f'Bridge entities are loaded from q{qname_entity} answers...')
            d_res_q, model_names, ids_paper = load_pred(f'q{qname_entity}',dir_pk=dir_output)
            d_res_q_pred, d_agg_q_pred = get_predictions(d_res_q, [],topk_cutoff=20,merge_answers=qmerge)
            
            df_q = pd.DataFrame(d_agg_q_pred)#[['id']+[x+'_answers' for x in models]]
            cols_combine = [c for c in df_q.columns if c.endswith('_answers')]
            
        df_q['combined_answers'] = df_q.apply(lambda x: list(set(list(itertools.chain.from_iterable(x[cols_combine].values.tolist())))),axis=1)
        df_q = df_q[['id','combined_answers']]
        df_q['len'] = df_q['combined_answers'].apply(lambda x: len(x))
        df_data = df_q.merge(df_text[['id','text_cleaned']],how='right',on='id')

        fn_log = f'inference_q{qname}.log' # this file logs the completed inference
        t_s_total = time.time()
        
        l_combinations_to_infer = [(qname,k) for k in qmodels]
        print(f'{len(l_combinations_to_infer)} model inference runs')
        try:
            with open(os.path.join(dir_output,fn_log)) as f:
                log = [l.split('\t') for l in f.readlines()]
            for qname,k,_ in log:
                l_combinations_to_infer.remove((qname,k))
            print(f'{len(log)} files are found in the log, removing from the config for inference...')
        except Exception as e:
            #print(e)
            pass
        print(f'{len(l_combinations_to_infer)} final model inference runs')
        
        print('*'*20,f'q{qname}',q,'*'*20)
        d_res = {}
        t_s_q = time.time()
        for k in qmodels: #k,v in d_models.items():
            count_q = 0
            if (qname,k) in l_combinations_to_infer:
                t_s_q_k = time.time()
                pipeline_qa = pipeline(task="question-answering", 
                                       model=k,
                                      device=device)
                d_res[k] = {}
                print(f'inference with model {k}...')
                
                for i in df_data.index:
                    context = df_data['text_cleaned'].loc[i]
                    id_paper = df_data['id'].loc[i]
                    entities = df_data['combined_answers'].loc[i]
                    d_res[k][id_paper] = []
                    for entity in entities:
                        entity = entity.strip()
                        if len(entity) > 0:
                            try:
                                d_res[k][id_paper].append({entity: pipeline_qa(question=q.replace(TOKEN_PLACEHOLDER, entity), context=context, top_k=5, device=device,batch_size=128,doc_stride=128,)})
                                count_q += 1
            
                                if count_q % 100 == 0:
                                    print(f'Done {count_q+1} questions')
                            except Exception as e:
                                print(e)
                                print(i,id_paper,q.replace(TOKEN_PLACEHOLDER, entity))
                                print(context)
                    
                    if i % 100 == 0 or i+1 == df_data.shape[0]:
                        print(f'Done {i+1}/{df_data.shape[0]}')
                        print(len(d_res),len(d_res[k]))
                time_used = time.time() - t_s_q_k
                print(f'Finished model {k} on q{qname}, elapsed time', time_used)
                pickle.dump(d_res[k],open(os.path.join(dir_output,f'd_res_q{qname}|{k.replace("/","__")}.pk'),'wb'))
    
                # update log file
                with open(os.path.join(dir_output,fn_log),'a') as f:
                    f.write('\t'.join([qname,k,str(time_used)])+'\n')
            else:
                print(f'Skipped model {k} on q{qname}: found record in log file')
                
        print(f'Finished all models on q{qname}, elapsed time', time.time() - t_s_q)
    print(f'Finished all models on all questions, elapsed time', time.time() - t_s_total)
    print()
    
    # remove pipeline_tg and release resource
    del pipeline_qa
    gc.collect()
    torch.cuda.empty_cache()

    #-------- Step 4 Prepare Final Output --------
    print('* Preparing final output...')
    df_res = pd.DataFrame()
    l_qname = []
    
    for qname, qconfig in d_qs.items():
        qtext = qconfig[0]
        qmodels = qconfig[1]
        qsetting = qconfig[2]
        
        if len(qsetting) > 0:
            if qsetting[0] == 'rag':
                df_q = pickle.load(open(os.path.join(dir_output,f'df_q{qname}_category.pk'),'rb'))
                cols_combine = [c for c in df_q.columns if c.endswith('_answers_categories')]
                df_q[f'q{qname}_combined_answers'] = df_q.apply(lambda x: list(set(list(itertools.chain.from_iterable(x[cols_combine].values.tolist())))),axis=1)
                l_qname.append(qname)
            
            elif qsetting[0] == 'msh':
                topk = 3 # the number of top answers to retain for each sub-question
                
                d_res_q, model_names, ids_paper = load_pred(f'q{qname}',dir_pk=dir_output)
                d_res_q_pred, d_agg_q_pred = get_predictions_msh(d_res_q, merge_all_answers=False, topk_cutoff=3)
                
                df_q = msh_to_df(d_res_q_pred)
                df_q.to_csv(os.path.join(dir_output,f'q{qname}_output_msh.csv'),index=False)
                continue
        else:
            d_res_q, model_names, ids_paper = load_pred(f'q{qname}',dir_pk=dir_output)
            d_res_q_pred, d_agg_q_pred = get_predictions(d_res_q, [],topk_cutoff=5)
            df_q = pd.DataFrame(d_agg_q_pred)
            
            cols_combine = [c for c in df_q.columns if c.endswith('_answers')]
            df_q[f'q{qname}_combined_answers'] = df_q.apply(lambda x: list(set(list(itertools.chain.from_iterable(x[cols_combine].values.tolist())))),axis=1)
            l_qname.append(qname)
            
        try:
            df_res = pd.merge(df_res,df_q[['id',f'q{qname}_combined_answers']],how='outer',on='id')
        except:
            df_res = df_q[['id',f'q{qname}_combined_answers']].copy()
        

    df_res.to_csv(os.path.join(dir_output,f"q{''.join(l_qname)}_output.csv"),index=False)
            
    
if __name__ == '__main__':
    
    print('* Loading source files...')
    try:
        df_text = pickle.load(open(path_text,'rb'))
    except:
        try:
            df_text = pd.read_csv(path_text)
        except:
            try:
                df_text = pd.read_excel(path_text)
            except:
                raise Exception('The file path cannot be opened by pickle.load(), pd.read_csv(), or pd.read_excel()')
    print(df_text)
    d_qs = json.load(open(path_questions))
    print(f'Questions loaded from {path_questions}:')
    pprint(d_qs)
    
    # verify config file
    for qname, qconfig in d_qs.items():
        assert len(qconfig) == 3, f"Wrong question config for {qconfig}: there should be 3 elements for each question name ({len(qconfig)} found)"
        qtext = qconfig[0]
        qmodels = qconfig[1]
        qsetting = qconfig[2]
        assert isinstance(qtext,str), f"Wrong question config for {qconfig}: the first element should be a string of question text ({qtext} found)"
        assert len(qsetting) == 0 or len(qsetting) >= 2, f"Wrong question config for {qconfig}: there should be either no setting (empty list) or a setting with at least name and information as two elements"
        if len(qsetting) > 0:
            assert qsetting[0] in ['rag','msh'], f"Wrong question config for {qconfig}: name {qsetting[0]} is not implemented, use either rag or msh"
            if qsetting[0] == 'rag':
                assert isinstance(qsetting[1],str), f"Wrong question config for {qconfig}: config setting for {qsetting[0]} should be string"
            elif qsetting[0] == 'msh':
                assert TOKEN_PLACEHOLDER in qtext, f"Wrong question config for {qconfig}: placeholder token {TOKEN_PLACEHOLDER} not found in question text ({qtext})"
    
    d_qid2qname = {i:k for i,k in enumerate(d_qs.keys())}
    d_qname2qid = {k:i for i,k in enumerate(d_qs.keys())}
    print()

    main()