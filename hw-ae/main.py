
from utils import parse_option, set_gpu, PrepareFunc, set_seeds, cal_debias_al_acc, debias_dataloader2tensor
import sys
import datetime
from pprint import pprint
import torch
import numpy as np

YOUR_STUDENT_ID = "TODO"

def train_rbm(model, train_loader, rbm_models, criterion, args):
    print("Begin training..")
    for epoch in range(args.max_epoch):
        epoch_loss = 0
        for idx, (x, _) in enumerate(train_loader):
            x = x.view(x.shape[0], -1).to(torch.device('cuda'))

            model.contrastive_divergence(rbm_models.v2h(x), args.lr_rbm)
            loss = criterion("TODO", x)
            epoch_loss += loss
        print(f'Epoch {epoch} Loss: {epoch_loss:.4f}.')
    print("Completed.")

def validate_loaded_ae(ae_model, rbm_models, train_loader):
    for idx, (x, _) in enumerate(train_loader):
        x = x.view(x.shape[0], -1).to(torch.device('cuda'))
        print(torch.norm(rbm_models.v2h2v(x) - ae_model(x)))

def train_ae(model, train_loader, criterion, optimizer, args):
    for epoch in range(args.max_epoch):
        epoch_loss = 0
        for idx, (x, _) in enumerate(train_loader):
            x = x.view(x.shape[0], -1).to(torch.device('cuda'))

            loss = "TODO (这里大约要写五行)"

            epoch_loss += loss
        print(f'Epoch {epoch} Loss: {epoch_loss:.4f}.')

def val_ae(model, test_loader, save_result=None, is_raw=False):
    from sklearn.linear_model import LogisticRegression
    hidden, label = [], []
    for idx, (x, y) in enumerate(test_loader):
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        if isinstance(y, list):
            y = y[0]
        x = x.view(x.shape[0], -1).to(torch.device('cuda'))
        if is_raw:
            hidden.append(x.detach().cpu())
        else:
            hidden.append(model.encoder(x).detach().cpu())
        label.append(y.cpu())
    hidden_np = torch.cat(hidden).numpy()
    label_np = torch.cat(label).numpy()
    clf = LogisticRegression()
    clf.fit(hidden_np, label_np)
    test_acc = clf.score(hidden_np, label_np)
    print(f'Test Accuracy: {test_acc}.')
    if save_result is not None:
        with open(f'{YOUR_STUDENT_ID}.csv', 'w') as f:
            f.write(f'{save_result},{test_acc}\n')

def tsne_ae(model, cur_loader, file_name='', is_raw=False):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn import manifold
    Axes3D

    sampled_num = 10 * 200
    hidden, label = [], []
    for idx, (x, y) in enumerate(cur_loader):
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        if isinstance(y, list):
            y = y[0]
        x = x.view(x.shape[0], -1).to(torch.device('cuda'))
        if is_raw:
            hidden.append(x.detach().cpu())
        else:
            hidden.append(model.encoder(x).detach().cpu())
        label.append(y.cpu())
    hidden_np = torch.cat(hidden).numpy()
    label_np = torch.cat(label).numpy()
    sampled_idx = np.random.choice(hidden_np.shape[0], sampled_num, replace=False)
    X, y = hidden_np[sampled_idx], label_np[sampled_idx]
    t_SNE_method = manifold.TSNE(n_components=2, init='pca', random_state=929)
    trans_X = t_SNE_method.fit_transform(X)
    plt.scatter(trans_X[:, 0], trans_X[:, 1], s=15, c=y, alpha=.4)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'saves/t-sne-{file_name}.png')
    plt.clf()

def val_extract_bias_conflicting(model, train_loader):
    hidden, label, label_bias, query_label = debias_dataloader2tensor(model, train_loader, category=1, is_raw=True)

    # from sklearn.svm import SVC
    # clf = SVC(kernel='rbf', class_weight='balanced', C=1.0)
    # clf.fit(hidden.numpy(), query_label.numpy())
    # query_idx = torch.arange(hidden.shape[0])[clf.predict(hidden.numpy())]
    # print(f'Test Accuracy: {cal_debias_al_acc(query_idx, label, label_bias)}.')

    from pyod.models.knn import KNN
    clf_name = 'KNN'
    clf = KNN(contamination=0.05)
    clf.fit(hidden.numpy())
    y_test_pred, y_test_pred_confidence = clf.predict(hidden.numpy(), return_confidence=True)

    query_idx = torch.arange(hidden.shape[0])[y_test_pred]
    print(f'Test Accuracy: {cal_debias_al_acc(query_idx, label, label_bias)}.')


if __name__ == '__main__':
    set_seeds(929, 929, 929, 929)
    is_colab = 'google.colab' in sys.modules
    args = parse_option()

    if args.time_str == '':
        args.time_str = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
    if not is_colab:
        set_gpu(args.gpu)
    pprint(vars(args))

    prepare_handle = PrepareFunc(args)
    train_loader, test_loader = prepare_handle.prepare_dataloader(args.dataset)

    ae_dims = [784, 2000, 1000, 500, 30]
    rbm_models = prepare_handle.prepare_model('rbm_handle')
    criterion = prepare_handle.prepare_loss_fn()
    if args.do_train_rbm:
        for in_features, out_features in zip(ae_dims[:-1], ae_dims[1:]):
            cur_model = prepare_handle.prepare_model('rbm', [in_features, out_features])
            train_rbm(cur_model, train_loader, rbm_models, criterion, args)
            rbm_models.append(cur_model)

    ae_model = prepare_handle.prepare_model('ae', ae_dims)

    prepare_handle.load_rbm_pretrained_models(ae_model, rbm_models)
    optimizer = prepare_handle.prepare_optimizer(ae_model)
    train_ae(ae_model, train_loader, criterion, optimizer, args)
    val_ae(ae_model, test_loader, save_result=True)
    tsne_ae(ae_model, test_loader, 'mnist-test')

    if args.bonus:
        debias_train_loader, debias_test_loader = prepare_handle.prepare_dataloader('ColoredMNIST')
        val_ae(ae_model, debias_test_loader)
        tsne_ae(ae_model, debias_test_loader, 'colored-mnist-test')

        val_extract_bias_conflicting(ae_model, debias_train_loader)

