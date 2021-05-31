# QHTE
Query-based Heterogeneous Treatment Effect estimation (QHTE), as developed by Qin, Wang and Zhou (2021), is a method that actively selects $(x,t)$ pairs to query for outcomes $y$ to better estimate Heterogeneous Treatment Effect (HTE) on a given budget. QHTE is implemented in Python using PyTorch. See requirements.txt for a full list of required packages. 

# Code
Three main components of this project are the neural network model for estimating HTEs (code/cfr_net.py), the core-set query strategy as suggested in the paper (code/strategy.py), and the one to finish the full training and evaluating process (code/main.py).

Some hyperparameters can be set by modifying the config.ini file. Most of the parameters are easy to understand. Here we highlight some of them to avoid confusion of potential users.
```
# Specify the files to be used in the input directory, can either be '(start_idx, end_idx)' or 'all'.
use_input = 0, 10 
# Whether to use batchnorm during training. 
bn = True 
# Specify the GPU to be used.
cuda = 0
# The hyperparamter for weighing the IPM term.
gamma = 1 
# Specify the query strategy, can be 'core_set' or 'random'.
strategy = core_set 
# Specify the step size between budgets.
budget_per_turn = 200 
```

# Running
After creating a Python3 environment, run the following code to install required packages.
```
pip install -r requirements.txt
```
Run code/main.py by specifying the configuration file with the -p argument.
```
python main.py -p config.ini
```
Some training information will be displayed if the verbose parameter is set.
```
Training ACIC1.csv with budget = 500, n_train = 350, n_val = 150
epoch = 50, training loss = 103.53903198242188, val_loss = 986.018798828125
epoch = 100, training loss = 53.55400466918945, val_loss = 891.850830078125
epoch = 150, training loss = 23.22492027282715, val_loss = 782.899169921875
...
test set: treated_rmse = 2.1433064171429415 control_rmse = 1.9365001682839886 sqrt_pehe = 1.8787106989178168
...
```
After the training, the results are saved to the specified directory.
# Reference
Tian Qin, Tian-Zuo Wang and Zhi-Hua Zhou. [Budgeted Heterogeneous Treatment Effect Estimation](https://arxiv.org/abs/1606.03976), 38th International Conference on Machine Learning (ICML), 2021.
