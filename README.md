# TCR specificity

Hyperparameter optimization was done using optuna, parameter search is saved in HPO folder for each implementation.
Each implementation with best performing hyperparameter included in scripts.

### Some Remarks
1. There are 4 ProtTrans models: ProtBERT, ProtAlbert, ProtElectra, ProtXLNet, each of these models were trained on specific
hyperparameters. These hyperparameters were obtained after hyperparameter optimization the history of which is available
in utils/tuners. We used optuna to optimise hyperparameters.
2. The 4 FineTuned ProtTrans models are then obtained which are available in /models, since we cannot share the entire model due to size,
we provide scripts to obtain them, which would take maximum two hours to obtain (Apologies for that). 
3. The HPC used to train the models had different GPUs available, so some models were trained on different GPUs due to 
configuration issues faced in some transformer models (for example ELECTRA requires too much GPU RAM to load onto it), 
so these specifications can be located in .sbatch files.
4. The Models used to predict dataset on IMMREP dataset are available in /predictions, so if you have dataset with common labels as with the vocabulary of the models available here, you can follow the similar route.
5. There are in total 12 models in this repository, with three configuration Baseline, Classification and Embedding denoted by 0, 1 and 2 respectively.
for each of the four Transformers we used. 
6. For any questions or issues, please create an issue in the repository.