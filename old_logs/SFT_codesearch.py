
from opendelta import AdapterModel , ParallelAdapterModel , LoraModel , PrefixModel
import argparse
import logging
import os
import pprint
import torch
import numpy as np
from model import  SFTModel
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm
import torch.nn as nn
import transformers
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.functional import binary_cross_entropy , binary_cross_entropy_with_logits
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaTokenizer , RobertaModel , AutoModel , AutoTokenizer , AutoConfig , AutoModelForSeq2SeqLM)
from utilities import *
from sklearn.metrics import recall_score, precision_score, f1_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("name")

#os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'







def train(args, model, tokenizer):
    """ Train the model """
    #get training dataset
    train_dataset_code_search = TextDataset_code_search(tokenizer, args, args.train_data_file_CodeSearch)
    train_dataloader_code_search = DataLoader(train_dataset_code_search, sampler=RandomSampler(train_dataset_code_search), batch_size=args.train_batch_size,num_workers=4)
    


    eval_dataset_code_search = TextDataset_code_search(tokenizer, args, args.eval_data_file_CodeSearch)
    eval_dataloader_code_search = DataLoader(eval_dataset_code_search, sampler=SequentialSampler(eval_dataset_code_search), batch_size=args.eval_batch_size,num_workers=4)



    test_dataset_code_search = TextDataset_code_search(tokenizer, args, args.test_data_file_CodeSearch)
    test_dataloader_code_search = DataLoader(test_dataset_code_search, sampler=SequentialSampler(test_dataset_code_search), batch_size=args.train_batch_size,num_workers=4)
    

   
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader_code_search) * args.num_train_epochs)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset_code_search))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader_code_search)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.train()
    tr_num,tr_loss,best_mrr = 0,0,0 
    train_loss  , eval_MRR = [] , []
    for idx in range(args.num_train_epochs): 
        LOSSes = []
        for step,batch in enumerate(train_dataloader_code_search):
            #get inputs
            code_inputs = batch[0].to(args.device)    
            nl_inputs = batch[1].to(args.device)
            #get code and nl vectors
            code_vec = model(code_inputs=code_inputs)
            nl_vec = model(nl_inputs=nl_inputs)
            
            #calculate scores and loss
            scores = torch.einsum("ab,cb->ac",nl_vec,code_vec)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores*20, torch.arange(code_inputs.size(0), device=scores.device))
            
            #report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)%100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss = 0
                tr_num = 0
            
            #backward
            loss.backward()
            LOSSes.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
        train_loss.append(np.mean(LOSSes))
        #evaluate    
        logger.info("***** Running evaluation *****")
        eval_results_code_search = evaluate_code_search(args, model,eval_dataloader_code_search, eval_dataloader_code_search , eval_when_training=True)
        eval_MRR.append(eval_results_code_search['eval_mrr'])

        for key, value in eval_results_code_search.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if eval_results_code_search['eval_mrr']>best_mrr:
            best_mrr = eval_results_code_search['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)  
            logger.info("***** Running Test *****")

            test_results_code_search = evaluate_code_search(args, model, test_dataloader_code_search, test_dataloader_code_search , eval_when_training=False)
            for key, value in test_results_code_search.items():
                    logger.info("  %s = %s", key, round(value,4))                        

            
    return train_loss , eval_MRR


def evaluate_code_search(args, model, query_dataloader , code_dataloader ,eval_when_training=False):
   

    # Eval!
   
    logger.info("  Num queries = %d", len(code_dataloader.dataset))
    logger.info("  Num codes = %d", len(code_dataloader.dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    code_vecs = [] 
    nl_vecs = []
    for batch in query_dataloader:  
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu().numpy()) 

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)    
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())  
    #model.train()    
    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)
    
    scores = np.matmul(nl_vecs,code_vecs.T)
    
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    
    nl_urls = []
    code_urls = []
    for example in query_dataloader.dataset.examples:
        nl_urls.append(example.url)
        
    for example in code_dataloader.dataset.examples:
        code_urls.append(example.url)

    ranks = []
    for url, sort_id in zip(nl_urls,sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)

    result = {
        "eval_mrr":float(np.mean(ranks))
    }

    return result





def main():


    parser = argparse.ArgumentParser()


    parser.add_argument("--output_dir", default='./', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_data_file_CodeSearch", default="./datasets/code_search/train.jsonl", type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--eval_data_file_CodeSearch", default="./datasets/code_search/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file_CodeSearch", default="./datasets/code_search/test.jsonl", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default="", type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    parser.add_argument("--model_name_or_path", default='microsoft/unixcoder-base', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.") 
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--train_data_rate_code_search", default=1.0, type=float,
                        help="The size of the train dataset")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--local_rank', default=-1 ,type=int,
                        help="random seed for initialization")
 
    args = parser.parse_args()
    set_seed(seed=args.seed)
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1 
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    config =   AutoConfig.from_pretrained(args.model_name_or_path)
    #config.dropout = args.dropout

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path ,trust_remote_code=True)
    try : 
        model = AutoModel.from_pretrained(args.model_name_or_path,config=config , trust_remote_code=True)
    except : 
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)


    if "codet5p" in args.model_name_or_path.lower() :  
        patterns = []
        for i in range(24):
            patterns.append("encoder.block." + str(i)+ ".layer.0" )
            patterns.append("encoder.block." + str(i)+ ".layer.1" ) 
            
    elif "codellama" in args.model_name_or_path.lower() : 
        patterns = ["self_attn", "mlp"]
    elif  "deepseek" in args.model_name_or_path.lower() : 
        patterns = []
        for i in range(24):
            #patterns.append("layers." + str(i)+ ".self_attn" )
            #patterns.append("layers." + str(i)+ ".mlp" ) 
            
            layer_key = str(i)
                # For self-attention, add each projection twice.
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    patterns.append(f"layers.{layer_key}.self_attn.{proj}")
                    patterns.append(f"layers.{layer_key}.self_attn.{proj}")
                # For the mlp module, add it twice.
            patterns.append(f"layers.{layer_key}.mlp")
            patterns.append(f"layers.{layer_key}.mlp")
            
    else :
        patterns = ['attention', '[r](\d)+\.output']
    

    
    #delta_model = ParallelAdapterModel(backbone_model=model , modified_modules= patterns , bottleneck_dim= 64)
    #delta_model = AdapterModel(backbone_model=model,modified_modules= patterns , bottleneck_dim=64 )
    delta_model = LoraModel(backbone_model=model , lora_r = 16  ,modified_modules=["q_proj", "v_proj"] )  # specify modules in case of deepseek model 
    #delta_model = PrefixModel(backbone_model=model )
    delta_model.freeze_module(exclude=["deltas", "classifier" ])
    delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)

    
    
    if "codet5" in args.model_name_or_path.lower() :
        try :
            model.module  = model.module.encoder 
        except : 
            model  = model.encoder 
    

    model = SFTModel( model , config)

    model.to(args.device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel( model)


    train_loss , eval_mrr =  train(args, model, tokenizer)

      



   
       
if __name__ == "__main__":
    main()