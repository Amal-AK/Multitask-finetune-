
from torch.utils.data.distributed import Dataset 
from torch.utils.data import RandomSampler
import torch
import json
import random
import logging
import math
import torch
import os
import numpy as np
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.nn.parallel.scatter_gather import scatter_kwargs

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger("name")




## vulnerabilty detection data ------------------------------------------------------------------------------------------------


class InputFeatures_vul_detect( object) : 

    def __init__(self,
                   code_tokens,
                   code_ids,
                   label): 
        self.code_tokens = code_tokens
        self.code_ids =  code_ids
        self.label = label
        self.task = 0 #"vul_detection"



class TextDataset_vul_detect(Dataset):
    def __init__(self, tokenizer, args, file_path=None,is_test=None,lang=None):
        self.examples = []
        self.len_list = []
        

        logger.info("Preparing the vulnerability detection Dataset...\n")
        data=[]
            
        with open(file_path) as f:
            for line in f:
                line = json.loads(line.strip())
                js = {}
                code = ' '.join(line['func'].split())
                label = int(line['target'])
                js['code_tokens'] = code
                js['target'] = label
                data.append(js)

        if 'train' in file_path : 
            size =   int (args.train_data_rate_vul * len(data))
        else : 
            size =  len(data)
       
        for js in data[:size]:
            self.examples.append(convert_examples_to_features_vul_detect(js,tokenizer,args))
            self.len_list.append(len(data))
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i): 
        x , y , z = torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].label), torch.tensor(self.examples[i].task )
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].label), torch.tensor(self.examples[i].task ))
    



def convert_examples_to_features_vul_detect(js,tokenizer,args):
    
        code=''.join(js['code_tokens'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        code_tokens=tokenizer.tokenize(code)[:args.code_length-2]
        #code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        code_tokens = add_special_tokens(tokenizer, code_tokens)
        code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = args.code_length - len(code_ids)
        code_ids+=[tokenizer.pad_token_id]*padding_length
        return InputFeatures_vul_detect(code_tokens,code_ids,js['target'])


## Clone detection dataset -----------------------------------------------------------------------------------------------





class InputFeatures_clone_detect ( object) : 

    def __init__(self,
                   code_ids,
                   label): 
        
        self.code_ids = code_ids
        self.label = label
        self.task = 1 #"clone_detection"




class TextDataset_clone_detect(Dataset):
    def __init__(self, tokenizer, args, file_path=None,is_test=None,lang=None):
        self.examples = []
       
        logger.info("Preparing the clone detection Dataset...\n")
        url_to_code = {}
        with open(args.data_file_clone) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                code = ' '.join(js['func'].split())
                url_to_code[js['idx']] = code

        data = []


        with open(file_path) as f:
            for line in f:
                js = {}
                line = line.strip()
                url1, url2, label = line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue

                js['code1'] = url_to_code[url1]
                js['code2']= url_to_code[url2]
                js['label']= int(label)
                data.append(js)
        if 'train' in file_path : 
            size =   int (args.train_data_rate_clone * len(data))
      
        else : 
            size =  len(data)
        for js in data[:size]:
            
            self.examples.append(convert_examples_to_features_clone_detect(js,tokenizer,args))  





    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].label), torch.tensor(self.examples[i].task ))
    
    



def convert_examples_to_features_clone_detect(js,tokenizer,args):

        ids_args =  ((js['code1'], js['code2']))
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        result = tokenizer(*ids_args, padding="max_length", max_length=args.code_length, truncation='longest_first')
    

        return InputFeatures_clone_detect( result['input_ids'] , js['label'])






# code search ----------------------------------------------------------------------------------------------------------



class InputFeatures_code_search(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url
        self.task = 2 #"code_search"

        
def convert_examples_to_features_code_search(js,tokenizer,args):
    """convert examples to token ids"""
    code_length = 256
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:code_length -4]
    #code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_tokens = add_special_tokens(tokenizer, code_tokens)
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
    #nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_tokens = add_special_tokens(tokenizer, nl_tokens)
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures_code_search(code_tokens,code_ids,nl_tokens,nl_ids,js['url'] if "url" in js else js["retrieval_idx"])

class TextDataset_code_search(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        logger.info("Preparing the code search Dataset...\n")
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
            elif "codebase"in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js) 
        if 'train' in file_path : 
            size =   int (args.train_data_rate_code_search * len(data))
        else : 
            size =  len(data) 
        for js in data[:size]:
            self.examples.append(convert_examples_to_features_code_search(js,tokenizer,args))                      
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids), torch.tensor(self.examples[i].task ))
            








#--------------flaky tests -----------------------------------------------------------------------------------------------------

class InputFeatures_flakyTest( object) : 

    def __init__(self,
                   code_tokens,
                   code_ids,
                   label): 
        self.code_tokens = code_tokens
        self.code_ids =  code_ids
        self.label = label
        self.task = 3 #"flakiness_detect"



class TextDataset_flakyTest(Dataset):
    def __init__(self, tokenizer, args, file_path=None,is_test=None,lang=None):
        self.examples = []
        self.len_list = []
        

        logger.info("Preparing the flakeFlager Dataset...\n")
        data=[]
            
    
        with open(file_path) as f:
            data_list = json.load(f)
            for line in data_list:
                js = {}
                code = ' '.join(line['code'].split())
                label = int(line['label'])
                js['code_tokens'] = code
                js['label'] = label
                data.append(js)

        if 'train' in file_path : 
            size =   int (args.train_data_rate_flaky * len(data))
        else : 
            size =  len(data)
        for js in data[:size]:
            self.examples.append(convert_examples_to_features_flakyTest(js,tokenizer,args))
            self.len_list.append(len(data))
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].label), torch.tensor(self.examples[i].task ))
    



def convert_examples_to_features_flakyTest(js,tokenizer,args):
    
        code=''.join(js['code_tokens'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        code_tokens=tokenizer.tokenize(code)[:args.code_length-2]
        #code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        code_tokens = add_special_tokens(tokenizer, code_tokens)
        code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = args.code_length - len(code_ids)
        code_ids+=[tokenizer.pad_token_id]*padding_length
        return InputFeatures_flakyTest(code_tokens,code_ids,js['label'])

def add_special_tokens(tokenizer, tokens):
    # tokens: list[str] (already tokenized)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
    return tokenizer.convert_ids_to_tokens(input_ids)



# ----------------------------------------------------------------------------------------------------------------------------


class EarlyStopper:
    def __init__(self, patience=2, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False




class MultiTaskEarlyStopper:
    def __init__(self, tasks, patience=2):
        """
        Args:
            tasks (list): List of task names.
            patience (int): Number of consecutive increases allowed.
        """
        self.patience = patience
        # For each task, keep a counter of consecutive increases.
        self.counters = {task: 0 for task in tasks}

    def early_stop(self, validation_results):
        """
        Checks each task's most recent two validation losses.
        
        Args:
            validation_results (dict): A dictionary of validation results for all tasks 
                
        Returns:
            bool: True if any task has increased its loss for 'patience' consecutive epochs,
                  False otherwise.
        """
        # Iterate over each task
        for task, metrics in validation_results.items():
            
            try :
                losses = metrics["eval_loss"]
            except :
                continue 
            # Need at least two epochs to compare
            if len(losses) < 2:
                continue

            previous_loss = losses[-2]
            current_loss = losses[-1]

            # Check if the loss increased compared to the previous epoch
            if current_loss > previous_loss:
                self.counters[task] += 1
                if self.counters[task] >= self.patience:
                    return True
            else:
                # Reset counter if loss decreased or stayed the same
                self.counters[task] = 0

        return False



# others -------------------------------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def update_validation_results(eval_results , validation_results):
    task_name = eval_results['task']
    
    # Get metric keys dynamically
    metric_keys = eval_results.keys() - {'task'}

    # Initialize sub-dictionary 
    if task_name not in validation_results:
        validation_results[task_name] = {key: [] for key in metric_keys}
    
    # Append the results to the respective arrays
    for key in metric_keys:
        validation_results[task_name][key].append(eval_results[key])



def save_trainable_params(model, save_path):
    """
    Save all trainable parameters (requires_grad=True) into a dict {name: tensor}.
    """
    m = model.module if hasattr(model, "module") else model
    trainable = {name: p.detach().cpu() for name, p in m.named_parameters() if p.requires_grad}

    if not trainable:
        raise RuntimeError("No trainable parameters found. Did you freeze everything?")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(trainable, save_path)
    print(f"Saved {len(trainable)} trainable params -> {save_path}")
    return list(trainable.keys())[:20]  # return a sample of names for sanity



def save_task_gradient (model , task_name , task_gradients) : 
    grad_vector = []
    for n, p in model.encoder.named_parameters():
        if p.grad is not None:
            grad_vector.append(p.grad.view(-1))
    grad_vector = torch.cat(grad_vector)
    task_gradients[task_name] = grad_vector.cpu()
    return task_gradients




def update_task_similarities(task_gradients, similarities_dict):
    """
    - task_gradients : { "task1": grad_tensor, "task2": grad_tensor, ... }
    - similarities_dict : un dictionnaire où on accumule les similarités 
        { ("task1","task2"): [sim_à_un_step, sim_à_un_autre_step, ...], ... }
    
    Cette fonction met à jour similarities_dict et le renvoie.
    """
    tasks = list(task_gradients.keys())
    # Tri pour être sûr d'avoir un ordre de paires cohérent (facultatif)
    tasks.sort()

    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            task_i = tasks[i]
            task_j = tasks[j]
            
            grad_i = task_gradients[task_i]
            grad_j = task_gradients[task_j]
            
            # Cosine similarity. Si vous voulez Pearson:
            # from scipy.stats import pearsonr
            # corr_ij, _ = pearsonr(grad_i.cpu().numpy(), grad_j.cpu().numpy())
            # sim = corr_ij
            sim = F.cosine_similarity(grad_i.unsqueeze(0), grad_j.unsqueeze(0), dim=1).item()
            
            # Stocker la similarité dans similarities_dict
            pair_key = str(task_i) + "_"+ str(task_j)

            if pair_key not in similarities_dict:
                similarities_dict[pair_key] = []
            similarities_dict[pair_key].append(np.round(sim,4))
    
    return similarities_dict



class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset.examples) for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)
    






#------------------------------------------------------------------------------------------------


class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).cpu().numpy(), acc.cpu().numpy()


def depth_error(x_pred, x_output):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return (
        torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)
    ).item(), (
        torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)
    ).item()


def normal_error(x_pred, x_output):
    binary_mask = torch.sum(x_output, dim=1) != 0
    error = (
        torch.acos(
            torch.clamp(
                torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1
            )
        )
        .detach()
        .cpu()
        .numpy()
    )
    error = np.degrees(error)
    return (
        np.mean(error),
        np.median(error),
        np.mean(error < 11.25),
        np.mean(error < 22.5),
        np.mean(error < 30),
    )


# for calculating \Delta_m
delta_stats = [
    "mean iou",
    "pix acc",
    "abs err",
    "rel err",
    "mean",
    "median",
    "<11.25",
    "<22.5",
    "<30",
]
BASE = np.array(
    [0.3830, 0.6376, 0.6754, 0.2780, 25.01, 19.21, 0.3014, 0.5720, 0.6915]
)  # base results from CAGrad
SIGN = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1])
KK = np.ones(9) * -1


def delta_fn(a):
    return (KK ** SIGN * (a - BASE) / BASE).mean() * 100.0  # * 100 for percentage






