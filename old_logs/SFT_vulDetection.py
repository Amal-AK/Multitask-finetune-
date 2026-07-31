
from opendelta import AdapterModel , LoraModel , ParallelAdapterModel , PrefixModel
import argparse
import logging
import os
import pprint
import torch
import numpy as np
from model import SFTModel_binary_classification , TASKS_STR
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader, SequentialSampler , RandomSampler
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AutoConfig , AutoTokenizer , AutoModel,   AutoModelForSeq2SeqLM)
from utilities import *
from sklearn.metrics import recall_score, precision_score, f1_score
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("name")

#os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'





def train(args, model,  tokenizer ):
    """ Train the model """

    # train data for vul detection 
    train_dataset_vul_detect=TextDataset_vul_detect(tokenizer, args, args.train_data_file_vul)
  
    # define the batch simpler to retrun in each batch data from same task 
    train_dataloader = DataLoader(dataset=train_dataset_vul_detect,
                                         sampler=RandomSampler(train_dataset_vul_detect),
                                         batch_size=args.train_batch_size,
                                         shuffle=False,
                                         num_workers=4,pin_memory=True)
    
    # prepare validation data 
    eval_dataset_vul= TextDataset_vul_detect(tokenizer, args,args.eval_data_file_vul)
    eval_dataloader_vul = DataLoader(eval_dataset_vul  , sampler=SequentialSampler(eval_dataset_vul ), batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)
   

    # prepare test dataloaders 
    test_dataset_vul= TextDataset_vul_detect(tokenizer, args,args.test_data_file_vul)
    test_dataloader_vul = DataLoader(test_dataset_vul  , sampler=SequentialSampler(test_dataset_vul ), batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)
   

    # define optimizer hyperparameters 
    optimizer =torch.optim.Adam(model.parameters(), lr=args.learning_rate )
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps*0.1, num_training_steps=max_steps)
    
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)


    logger.info("***** Running training *****")
    logger.info("  Num examples Vulnerabilty detection = %d", len(train_dataset_vul_detect))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size  = %d", args.train_batch_size)

    # initialisation 
    best_perfomance= - np.inf
    loss_fn = nn.BCELoss()
    #early_stopper = EarlyStopper(patience=3, min_delta=0.1)
    train_results =  {}
    validation_results = {}

    # epochs loop 
    model.zero_grad()

    for idx in range(args.num_train_epochs): 
        print("-"*150)
        global_acc = {}
        LOSSes   = [] 
        
        #bar = tqdm(train_dataloader,total=len(train_dataloader))
        #for step, batch in enumerate(bar):
        
        for step,batch in enumerate(train_dataloader): 
           

            model.train()


            #task 1 

            code_inputs = batch[0].to(args.device)  
            labels =  batch[1].to(args.device)  
            labels= labels.float().squeeze()
            logits = model(code_inputs=code_inputs).to(args.device)
            loss = loss_fn(logits,labels)
            accuracy = (logits.round() == labels ).float().mean().item()*100.0

            # perfom a backward step 
            LOSSes.append(loss.item() )
       
            # add current accuracies to accuracy arrays 
            global_acc.setdefault(batch[2][0], []).append(accuracy)
            
            # Calculate the mean accuracy for each task in the global_acc dictionary
            mean_accuracies = {task: np.mean(acc) for task, acc in global_acc.items()}

             # update progress bar
            #bar.set_description("Epoch {} Train Loss {}    ".format(idx, round(np.mean(LOSSes), 3)) +
                                #"  ".join(f"Acc {task}: {round(mean_acc, 3)}" for task, mean_acc in mean_accuracies.items()))

            if (step)%32 == 0:
                #torch.cuda.empty_cache()
                logger.info("Epoch {} Step {} Total Loss {}   ".format(idx ,step, round(np.mean(LOSSes), 3) ))

            
            loss.backward()
            
            # optimizer step 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
        

        train_results.setdefault('total_train_loss', []).append(round(np.mean(LOSSes),3))
     

        for key , value in global_acc.items():
                train_results.setdefault('train_acc_'+ TASKS_STR[key.item()], []).append(round(np.mean(value),3))

        # run a validation step for both tasks 
        eval_results = evaluate(args, model, eval_dataloader_vul  )

        # save validation results 
        update_validation_results(eval_results , validation_results)
       

        perfomance = eval_results['eval_acc']

        # Logging results for Task 1
        logger.info("\n***** Task 1 Evaluation Results *****")
        for key, value in eval_results.items():
            logger.info("  %s = %s", key, value )



        if perfomance >= best_perfomance  : 
            best_perfomance = perfomance
            save_best_model(model, args , checkpoint_prefix="models/deepseek_parallelAdapter_vul")
            logger.info("\n***** Running Test *****" ,)
            logger.info("  Num examples for vulnerability detection = %d", len(test_dataset_vul))
            logger.info("  Batch size = %d", args.eval_batch_size)

            test_result = test(args, model, test_dataloader_vul ) 
    
            

        #if early_stopper.early_stop(round(eval_results_task1['eval_loss'], 3)):             
            #break
        
    test_final = test(args, model, test_dataloader_vul ) 
  
        
    save_best_model(model, args , checkpoint_prefix="models/final_model_vul")
            

    return train_results , validation_results







# run validation for both tasks 
def evaluate(args, model, eval_dataloader_vul ):
        
        logger.info("\n***** Running evaluation *****")
        logger.info("  Num examples vulnerability detection = %d", len(eval_dataloader_vul.dataset))
        logger.info("  Batch size = %d ", args.eval_batch_size)

        model.eval()
        loss_fn = nn.BCELoss()

        eval_loss = 0.0
        nb_eval_steps = 0
        logits = []
        labels = []

        for batch in eval_dataloader_vul:
            inputs = batch[0].to(args.device)
            label = batch[1].to(args.device)
            with torch.no_grad():
                logit = model(code_inputs=inputs)
                label = label.float().squeeze()
                lm_loss = loss_fn(logit, label)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())
            nb_eval_steps += 1

        logits = np.concatenate(logits, 0)
        labels = np.concatenate(labels, 0)
        preds = logits.round()
        eval_acc = np.mean(labels ==  preds)
        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.tensor(eval_loss)
        recall = recall_score(labels , preds)
        precision = precision_score(labels , preds , zero_division=0)
        f1 = f1_score(labels , preds)

        result = {
            "task" : "vul_detection",
            "eval_loss": round(float(perplexity),4),
            "eval_acc": round(eval_acc, 4),
            "f1_score" : round(f1, 4),
            "recall" : round(recall,4),
            "precision" : round(precision,4)}
   

        return result






# Run test for one task 

def test(args, model, test_dataloader):

    logits = []
    labels = []

    for batch in test_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            task_name =batch[2][0]
            logit = model(code_inputs=inputs)
            label = label.float().squeeze()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits.round()
    acc = np.mean(labels ==  preds)
    recall = recall_score(labels , preds)
    precision = precision_score(labels , preds , zero_division=0)
    f1 = f1_score(labels , preds)

    result = {
            "task" : task_name,
            "test_acc": round(acc, 4),
            "test_f1_score" : round(f1, 4),
            "test_recall" : round(recall,4),
            "test_precision" : round(precision,4)
        }
    print("\n***** Test Results for task ", task_name)
    print(result , "\n\n")

    return result








def main():



    parser = argparse.ArgumentParser()


    parser.add_argument("--output_dir", default='./', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_classes", default=1, type=int,
                        help="The number of classes for the classification model")
    parser.add_argument("--train_data_file_vul", default="./datasets/dataset_vulnerabilty/train.jsonl", type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--eval_data_file_vul", default="./datasets/dataset_vulnerabilty/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file_vul", default="./datasets/dataset_vulnerabilty/test.jsonl", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    
    parser.add_argument("--model_name_or_path", default='microsoft/unixcoder-base', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=512, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.") 
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--train_data_rate_vul", default=0.01, type= float,
                        help="Data size for train")

    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    
    parser.add_argument("--dropout", default=0.3, type=float,
                        help="The initial learning rate for Adam.")
    
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=123456,
                        help="random seed for initialization")
    parser.add_argument('--local_rank', default=-1 ,type=int,
                        help="random seed for initialization")
 
    
    args = parser.parse_args()
    set_seed(seed=args.seed)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1 
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    config =   AutoConfig.from_pretrained(args.model_name_or_path)
    config.dropout = args.dropout

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
    

    
    delta_model = ParallelAdapterModel(backbone_model=model , modified_modules= patterns , bottleneck_dim= 64)
    #delta_model = AdapterModel(backbone_model=model,modified_modules= patterns , bottleneck_dim=64 )
    #delta_model = LoraModel(backbone_model=model, modified_modules=["q_proj", "v_proj"] , lora_r = 16)
    #delta_model = PrefixModel(backbone_model=model)
    delta_model.freeze_module(exclude=["deltas" ])
    delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)

    
    if "codet5" in args.model_name_or_path.lower() :
        try :
            model.module  = model.module.encoder 
        except : 
            model  = model.encoder 
    
    model = SFTModel_binary_classification(model,config)

    model.to(args.device)

    if args.n_gpu > 1:
         model = torch.nn.DataParallel( model)


    if args.do_train:
        train_results , valid_results = train(args , model ,tokenizer)
        print("\n Train results : \n")
        pprint.pprint(train_results )
        print("\n Validation results : \n")
        pprint.pprint( valid_results )



    if args.do_eval:
        checkpoint_prefix = 'models/final_model_vul/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir) , strict=False)      

        eval_dataset_vul= TextDataset_vul_detect(tokenizer, args,args.eval_data_file_vul)
        eval_dataloader_vul = DataLoader(eval_dataset_vul  , sampler=SequentialSampler(eval_dataset_vul ), batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)
       
        result_task1= evaluate(args, model, eval_dataloader_vul  )

        logger.info("\n***** Eval results *****")
        for key , value in result_task1.items() : 
            logger.info("  %s = %s", key, str(value))
 
                    

        
    if args.do_test:
            checkpoint_prefix = 'models/best_model_vul/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir),  strict=False)    

            test_dataset_vul= TextDataset_vul_detect(tokenizer, args,args.test_data_file_vul)
            test_dataloader_vul = DataLoader(test_dataset_vul  , sampler=SequentialSampler(test_dataset_vul ), batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)
           
            task1_test_result = test(args, model, test_dataloader_vul ) 
       

   
       
if __name__ == "__main__":
    main()