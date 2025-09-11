MTLfinetune
===========

Code repository for the research paper “Beyond Single-Task Fine-Tuning: Evaluating PEFT
Strategies in Multi-Task Code Analysis”

Contents
--------
- **env.yml**  
  Conda environment specification listing all required Python packages.

- **datasets/**  
  Evaluation datasets include :   
  - Devign :  vulnerability detection  
  - AdvTest :  code search 
  - FlakeFlagger :  Test Flakiness detection  
  - BigBenchClone : clone detection 

- **MTL_*.py / MTG_*.py**  
  Multi-task training pipelines covering the 4 tasks.

- **SFT_*.py**  
  Single-task finetuning baselines for clone detection, code search, vulnerability detection, and flaky test prediction.

- **train_*.sh**  
  Shell scripts to reproduce all training runs (4-task MTL, per-task SFT, etc.).
