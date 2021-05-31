import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import configparser
import os
import torch

from cfr_net import CFRNet, WrappedDataLoader


def init(path=None):
    config = configparser.ConfigParser()
    if path is None:
        config['data'] = {'input_dir': './data/IBM',
                          'output_dir': './results',
                          'val_ratio': 0.3,
                          'n_test_samples': 10000,
                          'use_input': 'all',
                          }
        config['model'] = {'repre_layers': '[200,200,200]',
                           'pred_layers': '[100,100,100]',
                           'cuda': 0,
                           'bn': False
                           }
        config['loss'] = {'alpha': 1,
                          'eps': 1e-3,
                          'max_iter': 10
                          }
        config['training'] = {'max_epochs': 3000,
                              'min_lr': 1e-6 + 1e-7,
                              'train_batch_size': 1000,
                              'test_batch_size': 1000,
                              'optimizer': 'sgd',
                              'lr': 1e-3,
                              'weight_decay': 1e-4,
                              'momentum': 0.9,
                              'nesterov': True,
                              'verbose': 1,
                              'patience': 20,
                              'cooldown': 20
                              }
        config['query'] = {'strategy': 'random',
                           'n_init': 1000,
                           'n_query_per_turn': 1000,
                           'n_query_max': 20000,
                           'n_set_size': 1,
                           'use_phi': False
                           }
        config['log'] = {'n_epochs_print': 50}
    else:
        config.read('config.ini')
    return config


def get_inputs(config):
    files = os.listdir(config['data']['input_dir'])
    list.sort(files)
    if config['data']['use_input'] != 'all':
        s, e = eval(config['data']['use_input'])
        files = files[s:min(e, len(files))]
    return files


def get_test_loader(test_data, test_batch_size):
    test_X = test_data.iloc[:, 5:].values
    test_y0, test_y1 = test_data['mu0'].values, test_data['mu1'].values
    test_treated_dl = WrappedDataLoader(test_X, np.ones(test_X.shape[0]), test_y1, test_batch_size, False)
    test_control_dl = WrappedDataLoader(test_X, np.zeros(test_X.shape[0]), test_y0, test_batch_size, False)
    return test_treated_dl, test_control_dl


def get_train_loader(train_data, train_batch_size):
    t = train_data['treatment'] == 1
    train_all_treated_dl = WrappedDataLoader(train_data[t].iloc[:, 5:].values,
                                             t.values.nonzero()[0],
                                             np.ones(t.sum()),
                                             train_batch_size, False)
    t = train_data['treatment'] == 0
    train_all_control_dl = WrappedDataLoader(train_data[t].iloc[:, 5:].values,
                                             t.values.nonzero()[0],
                                             np.ones(t.sum()),
                                             train_batch_size, False)
    return train_all_treated_dl, train_all_control_dl


def get_budgets(n_init, n_query_per_turn, n_query_max):
    tmp = list(range(n_init + n_query_per_turn, n_query_max + 1,
                     n_query_per_turn)) if type(n_query_per_turn) == int else n_query_per_turn
    budgets = [n_init] + [k for k in tmp if n_init < k <= n_query_max]
    return budgets


def get_models(input_dim, config):
    lr = eval(config['training']['lr'])
    n_repre_layers = eval(config['model']['repre_layers'])
    n_pred_layers = eval(config['model']['pred_layers'])
    bn = eval(config['model']['bn'])
    model = CFRNet(input_dim, n_repre_layers, n_pred_layers, bn)
    weight_decay = eval(config['training']['weight_decay'])
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=eval(config['training']['momentum']),
                              nesterov=eval(config['training']['nesterov']), weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=10)
    return model, optimizer, scheduler


def compute_rmse(model, dl, device):
    model.eval()
    with torch.no_grad():
        criterion = nn.MSELoss(reduction='sum')
        mse = sum(criterion(model(xb.to(device), tb.to(device)), yb.to(device)) for xb, tb, yb in
                           dl) / dl.get_X_size()[0]
    return np.sqrt(mse.item())


def compute_sqrt_pehe(model, treated_dl, control_dl, device):
    model.eval()
    n_samples = treated_dl.get_X_size()[0]
    with torch.no_grad():
        criterion = nn.MSELoss(reduction='sum')
        mse_treated = sum(criterion(model(xb.to(device), tb.to(device)), yb.to(device)) for xb, tb, yb in
                           treated_dl) / n_samples
        mse_control = sum(criterion(model(xb.to(device), tb.to(device)), yb.to(device)) for xb, tb, yb in
                           control_dl) / n_samples
        pehe2 = sum(
            criterion(model(xy1[0].to(device), xy1[1].to(device)) - model(xy0[0].to(device), xy0[1].to(device)),
                      xy1[2].to(device) - xy0[2].to(device)) for xy1, xy0 in
            zip(treated_dl, control_dl)) / n_samples
    return np.sqrt(pehe2.item()), np.sqrt(mse_treated.item()), np.sqrt(mse_control.item())


def choose_new_idx(start, end, selected, length):
    return list(np.random.choice(list(set(range(start,end))-set(selected)),min(length,end-start-len(selected)),replace=False))


def save_cont_results(model, test_treated_dl, test_control_dl, device, file, results, predictions, num_data, output_path):
    sqrt_pehe, rmse_treated, rmse_control = compute_sqrt_pehe(model, test_treated_dl, test_control_dl, device)
    print('test set: treated_rmse = {} control_rmse = {} sqrt_pehe = {}'.format(rmse_treated, rmse_control, sqrt_pehe))
    results.append([file, num_data, sqrt_pehe, rmse_treated, rmse_control])
    pd.DataFrame(results, columns=['file_name', 'budget', 'sqrt_pehe', 'rmse_treated', 'rmse_control']). \
        to_csv(output_path + '/summary_' + str(file) + ('' if file.endswith('.csv') else '.csv'), index=False)

    test_pred_y1 = np.vstack(
        [model(xb.to(device), tb.to(device)).cpu().detach().numpy() for xb, tb, yb in
         test_treated_dl])
    test_pred_y0 = np.vstack(
        [model(xb.to(device), tb.to(device)).cpu().detach().numpy() for xb, tb, yb in
         test_control_dl])
    test_y1 = np.vstack([np.array(yb) for _, _, yb in test_treated_dl])
    test_y0 = np.vstack([np.array(yb) for _, _, yb in test_control_dl])
    predictions.append(
        pd.DataFrame(np.hstack((np.ones(test_y0.shape) * num_data, test_y0, test_y1, test_pred_y0, test_pred_y1)),
                     columns=['n_query', 'y0', 'y1', 'y0_hat', 'y1_hat']))
    pd.concat(predictions, ignore_index=True).to_csv(output_path + '/predictions_' + str(file) + ('' if file.endswith('.csv') else '.csv'), index=False)
