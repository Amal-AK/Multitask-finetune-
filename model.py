import torch
import torch.nn as nn
import torch.nn.functional as F


TASKS_STR= {
    0: "vul_detection",
    1: "clone_detection",
    2: "code_search" ,
    3: "flakiness_detect"
}
#----------------------------------------------------------------------------------------



class  MultiTaskModel_MTL(nn.Module):   
    def __init__(self, encoder , config ):
        super(MultiTaskModel_MTL, self).__init__()
        self.encoder = encoder
        self.config = config 
        
        hs = getattr(config, "hidden_size", None)
        if hs is None:
            hs = getattr(config, "d_model", None)

        # CodeT5p-2B: nested encoder config
        if hs is None and hasattr(config, "encoder"):
            enc_cfg = getattr(config, "encoder")
            # enc_cfg may be a dict or an object
            hs = getattr(enc_cfg, "n_embd", None) if not isinstance(enc_cfg, dict) else enc_cfg.get("n_embd", None)

        # last-resort: read from actual encoder module
        if hs is None:
            enc_mod = getattr(self.encoder, "encoder", self.encoder)
            if hasattr(enc_mod, "wte"):                         # CodeT5p encoder
                hs = enc_mod.wte.weight.shape[1]
            elif hasattr(enc_mod, "embed_tokens"):              # T5-style
                hs = enc_mod.embed_tokens.weight.shape[1]

        if hs is None:
            raise AttributeError("Could not infer encoder hidden size (set config.d_model or encoder.n_embd).")

        self.hidden_size = hs
        self.task_weights = nn.Parameter(torch.zeros(len(config.tasks)))
        self.task_heads = nn.ModuleDict()
        
        for task in config.tasks:
            if task == "vul_detection":
                # Vulnerability detection head
                self.task_heads[task] = nn.Sequential(
                        nn.Linear(self.hidden_size, self.hidden_size // 2),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(self.hidden_size // 2, 1),
                    )
            elif task =="clone_detection" : 
                
                # Clone detection head
                self.task_heads[task] = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(p=0.1),
                    nn.Linear(self.hidden_size // 2, 1),
                    )
            elif task == "code_search" : 
                self.task_heads[task] = nn.Sequential(
                nn.Linear(self.hidden_size, 512),)
                
            elif task == "flakiness_detect" : 
                self.task_heads[task] = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(self.hidden_size // 2, 1),
                )
            else:
                raise ValueError(f"Unknown task: {task}")


        
    
    def forward(self, code_inputs=None, nl_inputs=None, task=None ):  
        if task is None:
            raise ValueError("A task must be specified for the forward pass.")
        
        
        if code_inputs is not None:
            
            if any(name in self.config._name_or_path.lower() for name in ["codet5", "codellama" , "deepseek" , "qwen"]):
                pad_id = self.config.pad_token_id if getattr(self.config, "pad_token_id", None) is not None else 0
                attention_mask = code_inputs.ne(pad_id) 
                outputs = self.encoder(code_inputs,attention_mask=attention_mask)
                outputs = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.size()).float()
                sum_embeddings = torch.sum(outputs * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                outputs = sum_embeddings / sum_mask
            elif  "microsoft" in self.config._name_or_path.lower() :
                outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[1]
            else : 
                raise ValueError("This model architecture is not supported yet")
              
              
                
        elif nl_inputs is not None : 
            
            if any(name in self.config._name_or_path.lower() for name in ["codet5", "codellama" , "deepseek" , "qwen"]):
                pad_id = self.config.pad_token_id if getattr(self.config, "pad_token_id", None) is not None else 0
                attention_mask = nl_inputs.ne(pad_id)
                outputs = self.encoder(nl_inputs,attention_mask=attention_mask)
                outputs = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.size()).float()
                sum_embeddings = torch.sum(outputs * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                outputs = sum_embeddings / sum_mask
            
            elif "microsoft" in self.config._name_or_path.lower() : 
                outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]
            else : 
                raise ValueError("This model architecture is not supported yet")
        
        
        else :
            raise ValueError("Inputs could not be None")
            
        if isinstance(task, int) : 
            task = TASKS_STR[task]
            head = self.task_heads[task]
            outs = head(outputs)
            if task == "code_search": 
                    outs = torch.nn.functional.normalize(outs, p=2, dim=1)
                    
        else : 
            
            unique_tasks = torch.unique(task) 
            predictions = []
            
            for t in unique_tasks:
                # Get indices for the current task t in the sub-batch
                indices = (task == t).nonzero(as_tuple=True)[0]
                # Select the pooled outputs corresponding to these indices
                task_outputs = outputs[indices]
                
                # Convert t (an int) to the task name string
                task_name = TASKS_STR[t.item()]
                head = self.task_heads[task_name]
                pred = head(task_outputs)
                if task_name == "code_search":
                    pred = torch.nn.functional.normalize(pred, p=2, dim=1)
                predictions.append(pred)
                
            outs =   torch.cat (predictions  , dim=0) 
            
        return outs 
    
    
    
    def get_task_weights(self):
        return F.softmax(self.task_weights , dim=0)
    



