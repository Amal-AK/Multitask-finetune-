import torch.nn as nn 
import torch

TASKS_STR= {
    0: "vul_detection",
    1: "clone_detection",
    2: "code_search" ,
    3: "flakiness_detect"
}
class  SFTModel(nn.Module):   
    def __init__(self, encoder , config ):
        super(SFTModel, self).__init__()
        self.encoder = encoder
        self.hidden_size = config.hidden_size
        self.config  =config
        self.projection =  nn.Linear(self.hidden_size, 512)

       
    
    def forward(self, code_inputs=None, nl_inputs=None, task=None): 
        
        if code_inputs is not None:
            
            if any(name in self.config._name_or_path.lower() for name in ["codet5", "codellama" , "deepseek"]):
                attention_mask =code_inputs.ne(1) 
                outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))
                outputs = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.size()).float()
                sum_embeddings = torch.sum(outputs * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                outputs = sum_embeddings / sum_mask
                
            elif "microsoft" in self.config._name_or_path.lower() : 
                outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[1]
            else : 
                raise ValueError("This model architecture is not supported yet")
                
        elif nl_inputs is not None : 
            
            if any(name in self.config._name_or_path.lower() for name in ["codet5", "codellama" , "deepseek"]):
                
                attention_mask =nl_inputs.ne(1) 
                outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))
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
        
        outputs = self.projection(outputs)
        
        return torch.nn.functional.normalize(outputs, p=2, dim=1)



class SFTModel_binary_classification ( nn.Module) : 
    def __init__(self, encoder , config ):
        super(SFTModel_binary_classification, self).__init__()
        self.encoder= encoder 
        self.hidden_size = config.hidden_size
        self.config  =config
        self.dropout = config.dropout 

        self.classification_head =  nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()

        )

    def forward(self, code_inputs=None, nl_inputs=None): 
        
        
        if code_inputs is not None:
            
            if any(name in self.config._name_or_path.lower() for name in ["codet5", "codellama" , "deepseek"]):
                attention_mask =code_inputs.ne(1) 
                outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))
                outputs = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.size()).float()
                sum_embeddings = torch.sum(outputs * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                outputs = sum_embeddings / sum_mask
            elif "microsoft" in self.config._name_or_path.lower() : 
                outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[1]
            else : 
                raise ValueError("This model architecture is not supported yet")
        else :
            raise ValueError("Inputs could not be None")
        
        outputs = self.classification_head(outputs).squeeze()
        
        return outputs