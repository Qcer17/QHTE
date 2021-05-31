from cfr_net import WrappedDataLoader

import numpy as np
import torch


def update_core_set(train_data, queried_train_idx0, queried_train_idx1, train_batch_size, train_all_control_dl,
                    train_all_treated_dl, model, selected_set, device, n_new_train):
    train_treated_dl = WrappedDataLoader(train_data.iloc[queried_train_idx1, 5:].values,
                                         np.array(queried_train_idx1),
                                         np.empty(len(queried_train_idx1)),
                                         train_batch_size,
                                         False)

    train_control_dl = WrappedDataLoader(train_data.iloc[queried_train_idx0, 5:].values,
                                         np.array(queried_train_idx0),
                                         np.empty(len(queried_train_idx0)),
                                         train_batch_size,
                                         False)

    min_dist_all = torch.ones(train_data.shape[0], device=device)
    for all_data, queried_data in [(train_all_treated_dl, train_treated_dl),
                                   (train_all_control_dl, train_control_dl)]:
        for X, idx, _ in all_data:
            X = model.get_repre(X, device)
            min_dist = torch.ones(X.shape[0], device=device) * np.inf
            for qX, qidx, _ in queried_data:
                qX = model.get_repre(qX, device)
                dist_mat = torch.cdist(X.to(device), qX.to(device))
                min_dist = torch.minimum(min_dist, torch.min(dist_mat, dim=1)[0])
            min_dist_all[idx.squeeze().long()] = min_dist
    new_train_idx = []
    while True:
        new_core_idx = -1
        new_core_treatment = -1
        for idx in list(torch.argsort(min_dist_all))[::-1]:
            if idx.item() not in selected_set:
                new_core_idx = idx.item()
                new_core_treatment = train_data['treatment'].iloc[idx.item()]
                selected_set.add(idx.item())
                new_train_idx.append(idx.item())
                break
        if len(new_train_idx) == n_new_train:
            break

        new_core = model.get_repre(torch.from_numpy(train_data.values[[new_core_idx], 5:]).float(), device)
        for X, idx, _ in (train_all_control_dl if new_core_treatment == 0 else train_all_treated_dl):
            X = model.get_repre(X, device)
            dist_mat = torch.cdist(X.to(device), new_core.to(device))
            min_dist_all[idx.squeeze().long()] = torch.minimum(min_dist_all[idx.squeeze().long()], dist_mat[:, 0])
    return new_train_idx


def get_n_covered(train_data, queried_train_idx0, queried_train_idx1,
                                                      train_batch_size, train_all_control_dl, train_all_treated_dl,
                                                      model, device):
    train_treated_dl = WrappedDataLoader(train_data.iloc[queried_train_idx1, 5:].values,
                                         np.array(queried_train_idx1),
                                         np.empty(len(queried_train_idx1)),
                                         train_batch_size,
                                         False)

    train_control_dl = WrappedDataLoader(train_data.iloc[queried_train_idx0, 5:].values,
                                         np.array(queried_train_idx0),
                                         np.empty(len(queried_train_idx0)),
                                         train_batch_size,
                                         False)

    n_covered = torch.zeros(train_data.shape[0], device=device, dtype=int)
    n_covered[queried_train_idx0] += 1
    n_covered[queried_train_idx1] += 1
    for all_data, queried_data in [(train_all_treated_dl, train_treated_dl),
                                   (train_all_control_dl, train_control_dl)]:
        for X, idx, _ in all_data:
            X = model.get_repre(X, device)
            cur_min_dist = torch.ones(X.shape[0], device=device) * np.inf
            cur_nearest_core_idx = torch.ones(X.shape[0], device=device, dtype=int)*-1
            for qX, qidx, _ in queried_data:
                qX = model.get_repre(qX, device)
                dist_mat = torch.cdist(X.to(device), qX.to(device))
                min_dist, min_idx = torch.min(dist_mat, dim=1)
                nearest_core_idx = qidx[min_idx,:].squeeze().to(device).long()
                cur_nearest_core_idx = torch.where(cur_min_dist<min_dist, cur_nearest_core_idx, nearest_core_idx)
                cur_min_dist = torch.minimum(cur_min_dist, min_dist)
            n_covered += torch.bincount(cur_nearest_core_idx,  minlength=train_data.shape[0])
    n_covered = n_covered[n_covered>0]
    return n_covered.cpu().numpy()
