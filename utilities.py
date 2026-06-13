from torch.utils.data import Dataset, RandomSampler
import torch
import json
import random
import logging
import math
import os
import numpy as np
import torch.nn.functional as F

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger("name")




## vulnerabilty detection data ------------------------------------------------------------------------------------------------


class InputFeatures_vul_detect(object):

    def __init__(self, code_tokens, code_ids, label):
        self.code_tokens = code_tokens
        self.code_ids    = code_ids
        self.label       = label
        self.task        = 0  # "vul_detection"


class TextDataset_vul_detect(Dataset):
    def __init__(self, tokenizer, args, file_path=None, is_test=None, lang=None):
        self.examples = []

        logger.info("Preparing the vulnerability detection Dataset...\n")
        is_train = 'train' in file_path
        limit = getattr(args, 'max_train_samples', None) if is_train else getattr(args, 'max_eval_samples', None)

        data = []
        with open(file_path) as f:
            for line in f:
                line = json.loads(line.strip())
                data.append({
                    'code_tokens': ' '.join(line['func'].split()),
                    'target':      int(line['target']),
                })
                if limit and len(data) >= limit:
                    break

        if is_train and not limit:
            data = data[:int(args.train_data_rate_vul * len(data))]
        for js in data:
            self.examples.append(convert_examples_to_features_vul_detect(js, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i].code_ids),
            torch.tensor(self.examples[i].label),
            torch.tensor(self.examples[i].task),
        )


def convert_examples_to_features_vul_detect(js, tokenizer, args):
    code = ''.join(js['code_tokens'])
    enc  = tokenizer(code, max_length=args.code_length, padding='max_length', truncation=True)
    return InputFeatures_vul_detect(None, enc['input_ids'], js['target'])


## Clone detection dataset -----------------------------------------------------------------------------------------------


class InputFeatures_clone_detect(object):

    def __init__(self, code_ids, label):
        self.code_ids = code_ids
        self.label    = label
        self.task     = 1  # "clone_detection"


class TextDataset_clone_detect(Dataset):
    def __init__(self, tokenizer, args, file_path=None, is_test=None, lang=None):
        self.examples = []

        logger.info("Preparing the clone detection Dataset...\n")
        is_train = 'train' in file_path
        limit = getattr(args, 'max_train_samples', None) if is_train else getattr(args, 'max_eval_samples', None)

        # Full code pool needed to resolve any pair URL — must be read entirely
        url_to_code = {}
        with open(args.data_file_clone) as f:
            for line in f:
                js = json.loads(line.strip())
                url_to_code[js['idx']] = ' '.join(js['func'].split())

        data = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                url1, url2, label = line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                data.append({
                    'code1':  url_to_code[url1],
                    'code2':  url_to_code[url2],
                    'label':  int(label),
                })
                if limit and len(data) >= limit:
                    break

        if is_train and not limit:
            data = data[:int(args.train_data_rate_clone * len(data))]
        for js in data:
            self.examples.append(convert_examples_to_features_clone_detect(js, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i].code_ids),
            torch.tensor(self.examples[i].label),
            torch.tensor(self.examples[i].task),
        )


def convert_examples_to_features_clone_detect(js, tokenizer, args):
    result = tokenizer(
        js['code1'], js['code2'],
        padding='max_length', max_length=args.code_length, truncation='longest_first',
    )
    return InputFeatures_clone_detect(result['input_ids'], js['label'])


# code search ----------------------------------------------------------------------------------------------------------


class InputFeatures_code_search(object):
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, url):
        self.code_tokens = code_tokens
        self.code_ids    = code_ids
        self.nl_tokens   = nl_tokens
        self.nl_ids      = nl_ids
        self.url         = url
        self.task        = 2  # "code_search"


def convert_examples_to_features_code_search(js, tokenizer, args):
    code = ' '.join(js['code_tokens']) if isinstance(js['code_tokens'], list) \
           else ' '.join(js['code_tokens'].split())
    nl_raw = js.get('docstring_tokens') or js.get('doc', '')
    nl = ' '.join(nl_raw) if isinstance(nl_raw, list) else ' '.join(nl_raw.split())

    code_enc = tokenizer(code, max_length=args.code_length, padding='max_length', truncation=True)
    nl_enc   = tokenizer(nl,   max_length=args.nl_length,   padding='max_length', truncation=True)

    url = js['url'] if 'url' in js else js['retrieval_idx']
    return InputFeatures_code_search(None, code_enc['input_ids'], None, nl_enc['input_ids'], url)


class TextDataset_code_search(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        logger.info("Preparing the code search Dataset...\n")
        is_train = 'train' in file_path
        limit = getattr(args, 'max_train_samples', None) if is_train else getattr(args, 'max_eval_samples', None)

        data = []
        with open(file_path) as f:
            if 'jsonl' in file_path:
                for line in f:
                    js = json.loads(line.strip())
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
                    if limit and len(data) >= limit:
                        break
            elif 'codebase' in file_path or 'code_idx_map' in file_path:
                raw = json.load(f)
                for key in raw:
                    data.append({
                        'code_tokens':    key.split(),
                        'retrieval_idx':  raw[key],
                        'doc':            '',
                        'docstring_tokens': '',
                    })
                    if limit and len(data) >= limit:
                        break
            elif 'json' in file_path:
                for js in json.load(f):
                    data.append(js)
                    if limit and len(data) >= limit:
                        break

        if is_train and not limit:
            data = data[:int(args.train_data_rate_code_search * len(data))]
        for js in data:
            self.examples.append(convert_examples_to_features_code_search(js, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i].code_ids),
            torch.tensor(self.examples[i].nl_ids),
            torch.tensor(self.examples[i].task),
        )


# flaky tests ----------------------------------------------------------------------------------------------------------


class InputFeatures_flakyTest(object):

    def __init__(self, code_tokens, code_ids, label):
        self.code_tokens = code_tokens
        self.code_ids    = code_ids
        self.label       = label
        self.task        = 3  # "flakiness_detect"


class TextDataset_flakyTest(Dataset):
    def __init__(self, tokenizer, args, file_path=None, is_test=None, lang=None):
        self.examples = []

        logger.info("Preparing the flakeFlager Dataset...\n")
        is_train = 'train' in file_path
        limit = getattr(args, 'max_train_samples', None) if is_train else getattr(args, 'max_eval_samples', None)

        data = []
        with open(file_path) as f:
            for line in json.load(f):
                data.append({
                    'code_tokens': ' '.join(line['code'].split()),
                    'label':       int(line['label']),
                })
                if limit and len(data) >= limit:
                    break

        if is_train and not limit:
            data = data[:int(args.train_data_rate_flaky * len(data))]
        for js in data:
            self.examples.append(convert_examples_to_features_flakyTest(js, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i].code_ids),
            torch.tensor(self.examples[i].label),
            torch.tensor(self.examples[i].task),
        )


def convert_examples_to_features_flakyTest(js, tokenizer, args):
    code = ''.join(js['code_tokens'])
    enc  = tokenizer(code, max_length=args.code_length, padding='max_length', truncation=True)
    return InputFeatures_flakyTest(None, enc['input_ids'], js['label'])


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
        self.patience = patience
        self.counters = {task: 0 for task in tasks}

    def early_stop(self, validation_results):
        for task, metrics in validation_results.items():
            try:
                losses = metrics["eval_loss"]
            except Exception:
                continue
            if len(losses) < 2:
                continue
            if losses[-1] > losses[-2]:
                self.counters[task] += 1
                if self.counters[task] >= self.patience:
                    return True
            else:
                self.counters[task] = 0
        return False


# others -------------------------------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # seeds every visible GPU, not just cuda:0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # benchmark=True overrides deterministic


def update_validation_results(eval_results, validation_results):
    task_name   = eval_results['task']
    metric_keys = eval_results.keys() - {'task'}

    if task_name not in validation_results:
        validation_results[task_name] = {key: [] for key in metric_keys}

    for key in metric_keys:
        validation_results[task_name][key].append(eval_results[key])


def save_trainable_params(model, save_path):
    """Save all trainable parameters (requires_grad=True) to a state-dict file."""
    m = model.module if hasattr(model, "module") else model
    trainable = {name: p.detach().cpu() for name, p in m.named_parameters() if p.requires_grad}

    if not trainable:
        raise RuntimeError("No trainable parameters found. Did you freeze everything?")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(trainable, save_path)
    print(f"Saved {len(trainable)} trainable params -> {save_path}")
    return list(trainable.keys())[:20]


def load_trainable_params(model, load_path):
    """Load trainable parameters saved by save_trainable_params back into model."""
    m = model.module if hasattr(model, "module") else model
    state = torch.load(load_path, map_location="cpu")
    current = dict(m.named_parameters())
    unexpected, missing = [], []

    for name, param in state.items():
        if name in current:
            current[name].data.copy_(param.to(current[name].device))
        else:
            unexpected.append(name)

    for name, p in current.items():
        if name not in state and p.requires_grad:
            missing.append(name)

    if unexpected:
        logger.warning("load_trainable_params: unexpected keys: %s", unexpected)
    if missing:
        logger.warning("load_trainable_params: missing trainable keys: %s", missing)

    print(f"Loaded {len(state) - len(unexpected)} params from {load_path}")


def save_task_gradient(model, task_name, task_gradients):
    m = model.module if hasattr(model, "module") else model
    grad_vector = []
    for n, p in m.encoder.named_parameters():
        if p.grad is not None:
            grad_vector.append(p.grad.view(-1))
    grad_vector = torch.cat(grad_vector)
    task_gradients[task_name] = grad_vector.cpu()
    return task_gradients


def update_task_similarities(task_gradients, similarities_dict):
    tasks = sorted(task_gradients.keys())
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            grad_i = task_gradients[tasks[i]]
            grad_j = task_gradients[tasks[j]]
            sim = F.cosine_similarity(grad_i.unsqueeze(0), grad_j.unsqueeze(0), dim=1).item()
            pair_key = f"{tasks[i]}_{tasks[j]}"
            similarities_dict.setdefault(pair_key, []).append(round(sim, 4))
    return similarities_dict


class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """Iterate over tasks and provide a random batch per task in each mini-batch."""
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max(len(d.examples) for d in dataset.datasets)

    def __len__(self):
        return (
            self.batch_size
            * math.ceil(self.largest_dataset_size / self.batch_size)
            * self.number_of_datasets
        )

    def __iter__(self):
        samplers_list     = [RandomSampler(self.dataset.datasets[i]) for i in range(self.number_of_datasets)]
        sampler_iterators = [iter(s) for s in samplers_list]
        push_index_val    = [0] + self.dataset.cumulative_sizes[:-1]

        step           = self.batch_size * self.number_of_datasets
        epoch_samples  = self.largest_dataset_size * self.number_of_datasets
        final_samples_list = []

        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_samples = []
                for _ in range(self.batch_size):
                    try:
                        idx = next(sampler_iterators[i])
                    except StopIteration:
                        sampler_iterators[i] = iter(samplers_list[i])
                        idx = next(sampler_iterators[i])
                    cur_samples.append(idx + push_index_val[i])
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


class TemperatureSampler(torch.utils.data.sampler.Sampler):
    """Sample task batches with probability proportional to N_i^T.

    At each step one task is drawn from a temperature-scaled multinomial, then
    a full batch is drawn from that task's shuffled index pool (cycling with a
    reshuffle when exhausted).  One epoch spans ceil(sum_i N_i / batch_size)
    steps — roughly one expected pass through each task, weighted by temperature.

    T → 0:  uniform across tasks (N_i^0 = 1 for all i).
    T = 0.5: mT5 style — prob proportional to sqrt(N_i); large datasets downscaled.
    T = 1:  proportional to dataset size (no correction).
    T > 1:  super-linear; largest dataset dominates progressively more.
    """
    def __init__(self, dataset, batch_size, temperature=0.5, seed=42):
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.temperature = temperature
        self.seed        = seed

        datasets         = dataset.datasets
        self.task_sizes  = [len(d.examples) for d in datasets]
        self.offsets     = [0] + list(dataset.cumulative_sizes[:-1])
        self.n_tasks     = len(datasets)

        sizes   = np.array(self.task_sizes, dtype=float)
        weights = sizes ** max(temperature, 1e-8)   # T=0.5 → sqrt(N_i); T=1 → proportional
        self.probs = (weights / weights.sum()).tolist()

        self.n_batches = math.ceil(sum(self.task_sizes) / batch_size)

    def __len__(self):
        return self.n_batches * self.batch_size

    def __iter__(self):
        rng          = np.random.default_rng(self.seed)
        task_indices = [list(torch.randperm(sz).numpy()) for sz in self.task_sizes]
        cursors      = [0] * self.n_tasks

        indices = []
        for _ in range(self.n_batches):
            t      = int(rng.choice(self.n_tasks, p=self.probs))
            offset = self.offsets[t]
            size   = self.task_sizes[t]
            for _ in range(self.batch_size):
                if cursors[t] >= size:
                    task_indices[t] = list(torch.randperm(size).numpy())
                    cursors[t]      = 0
                indices.append(offset + task_indices[t][cursors[t]])
                cursors[t] += 1

        return iter(indices)
