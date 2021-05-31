import time
import argparse

from utils import *
from loss import *
from strategy import *

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-p', '--path', type=str, nargs='+', dest='path',
                      default= 'config.ini',
                      help='The path of the configuration file.')
    args = args.parse_args()

    config = init(args.path)

    os.environ["CUDA_VISIBLE_DEVICES"] = config['model']['cuda']
    val_ratio = eval(config['data']['val_ratio'])
    n_test_sample = eval(config['data']['n_test_samples'])
    epochs = eval(config['training']['max_epochs'])
    train_batch_size = eval(config['training']['train_batch_size'])
    test_batch_size = eval(config['training']['test_batch_size'])
    min_lr = eval(config['training']['min_lr'])
    use_weight = eval(config['training']['use_weight'])
    verbose = eval(config['training']['verbose'])
    gamma = eval(config['loss']['gamma'])
    n_init = eval(config['query']['n_init'])
    budget_per_turn = eval(config['query']['budget_per_turn'])
    max_budget = eval(config['query']['max_budget'])
    n_epochs_print = eval(config['log']['n_epochs_print'])
    output_path = config['data']['output_dir'] + '/' + str(int(time.time()))
    os.mkdir(output_path)

    with open(output_path + '/config.ini', mode='w') as f:
        config.write(f)

    input_files = get_inputs(config)

    for file in input_files:
        results = []
        predictions = []

        # ------------------------ init ---------------------------------
        data = pd.read_csv(config['data']['input_dir'] + '/' + file).sample(frac=1).reset_index(drop=True)
        test_data = data.iloc[-n_test_sample:, :]
        train_data = data.iloc[:-n_test_sample, :]

        # construct test set and the set containing full training data
        test_treated_dl, test_control_dl = get_test_loader(test_data, test_batch_size)
        train_all_treated_dl, train_all_control_dl = get_train_loader(train_data, test_batch_size)

        # construct training and validation set by random sampling
        n_val = int(n_init * val_ratio)
        n_train = n_init - n_val
        queried_train_idx0 = list(
            np.random.choice((train_data['treatment'] == 0).values.nonzero()[0], n_train // 2, replace=False))
        queried_train_idx1 = list(
            np.random.choice((train_data['treatment'] == 1).values.nonzero()[0], n_train - n_train // 2, replace=False))
        queried_val_idx = choose_new_idx(0, train_data.shape[0], queried_train_idx0 + queried_train_idx1, n_val)
        n_covered = np.ones(n_train, dtype=int)

        # get increasing budgets
        budgets = get_budgets(n_init, budget_per_turn, min(train_data.shape[0], max_budget))

        for budget, next_budget in zip(budgets, budgets[1:] + [0]):
            queried_train_idx = queried_train_idx0 + queried_train_idx1
            assert len(queried_val_idx) + len(queried_train_idx) <= budget

            # ----------------------------- data ----------------------------------
            train_X, val_X = train_data.iloc[queried_train_idx, 5:].values, train_data.iloc[queried_val_idx, 5:].values
            train_y, val_y = train_data['yf'][queried_train_idx].values, data['yf'][queried_val_idx].values
            train_t, val_t = data['treatment'][queried_train_idx].values, data['treatment'][queried_val_idx].values
            train_dl = WrappedDataLoader(train_X, train_t, train_y, train_batch_size, n_covered=n_covered)
            val_dl = WrappedDataLoader(val_X, val_t, val_y, test_batch_size)

            if verbose > 0:
                print('Training {} with budget = {}, n_train = {}, n_val = {}'.
                      format(file, budget, train_X.shape[0], val_X.shape[0]))

            # ----------------------------- model ----------------------------------
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, optimizer, scheduler = get_models(train_X.shape[1], config)
            sinkhorn = SinkhornDistance(eps=eval(config['loss']['eps']),
                                        max_iter=eval(config['loss']['max_iter']),
                                        reduction='mean', device=device)
            model.to(device)

            for epoch in range(epochs):
                # ------------------------ training --------------------------------
                model.train()
                training_loss = 0
                for xb, tb, yb, wb in train_dl:
                    xb, tb, yb, wb = xb.to(device), tb.to(device), yb.to(device), wb.to(device)
                    pred = model(xb, tb)
                    loss = total_loss(pred, model.repre, tb, yb, wb, dist=sinkhorn, gamma=gamma)
                    training_loss += loss
                    loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), 0.5)
                    optimizer.step()
                    optimizer.zero_grad()
                training_loss /= len(train_dl)

                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for xb, tb, yb in val_dl:
                        xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
                        pred = model(xb, tb)
                        loss = total_loss(pred, model.repre, tb, yb, dist=sinkhorn, gamma=gamma)
                        val_loss += loss
                    val_loss /= len(val_dl)
                    scheduler.step(val_loss)

                    if verbose > 0 and (epoch + 1) % n_epochs_print == 0:
                        print(
                            'epoch = {}, training loss = {}, val_loss = {}'.format(epoch + 1, training_loss, val_loss))

                    if optimizer.state_dict()['param_groups'][0]['lr'] <= min_lr or epoch == epochs - 1:
                        # ------------------ save predictions to file ----------------
                        save_cont_results(model, test_treated_dl, test_control_dl, device, file, results,
                                          predictions, budget, output_path)
                        if budget == budgets[-1]:
                            break
                        # -------------------------- query ---------------------------
                        next_budget = min(next_budget, train_data.shape[0])
                        n_new_val = int(next_budget * val_ratio) - len(queried_val_idx)
                        n_new_train = next_budget - n_new_val - len(queried_train_idx) - len(queried_val_idx)
                        selected_set = set(queried_val_idx + queried_train_idx)
                        if config['query']['strategy'] == 'random':
                            new_train_idx = choose_new_idx(0, train_data.shape[0], queried_train_idx + queried_val_idx,
                                                           n_new_train)
                        elif config['query']['strategy'] == 'core_set':
                            new_train_idx = update_core_set(train_data, queried_train_idx0, queried_train_idx1,
                                                            train_batch_size, train_all_control_dl,
                                                            train_all_treated_dl,
                                                            model, selected_set, device, n_new_train)
                        else:
                            print('Unimplemented query strategy!')
                        for idx in new_train_idx:
                            if train_data['treatment'][idx] == 0:
                                queried_train_idx0.append(idx)
                            else:
                                queried_train_idx1.append(idx)
                        if config['query']['strategy'] == 'core_set' and use_weight:
                            n_covered = get_n_covered(train_data, queried_train_idx0, queried_train_idx1,
                                                      train_batch_size, train_all_control_dl, train_all_treated_dl,
                                                      model, device)
                        else:
                            n_covered = np.ones(len(new_train_idx) + len(queried_train_idx))
                        queried_val_idx += choose_new_idx(0, train_data.shape[0], queried_train_idx + queried_val_idx,
                                                          n_new_val)
                        break
