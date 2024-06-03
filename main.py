import argparse
import pickle
import time
import datetime
from torch.utils.data import DataLoader
from model import *
from util import *

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='Tmall', help='datasets name: Tmall/diginetica')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--emb_size', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-4, help='l2 penalty')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--sample_order', type=int, default=3)
parser.add_argument('--si_dc', type=float, default=2, help='session impact rate decay rate')
parser.add_argument('--is_train', type=bool, default=False)
parser.add_argument('--k', type=float, default=12)
parser.add_argument('--β', type=float, default=10000)
opt = parser.parse_args()

def main():
    init_seed(2024)
    if opt.datasets == 'diginetica':
        n_node = 43097
    elif opt.datasets == 'Tmall':
        n_node = 40727
    print(opt)

    print('start load data', datetime.datetime.now())
    if opt.is_train == True:
        with open(opt.datasets + '/train.txt', 'rb') as f:
            train_data = pickle.load(f)
        with open(opt.datasets + '/train_groups.txt', 'rb') as f:
            train_groups_data = pickle.load(f)
    with open(opt.datasets + '/test.txt', 'rb') as f:
        test_data = pickle.load(f)
    with open(opt.datasets + '/test_groups.txt', 'rb') as f:
        test_groups_data = pickle.load(f)
    print('end load data', datetime.datetime.now())

    if opt.is_train == True:
        train_data = Data(train_data, train_groups_data, opt.sample_order, opt.si_dc)
        train_loader = DataLoader(train_data, batch_size=opt.batch_size, pin_memory=True, shuffle=True, num_workers=8, collate_fn=collate_fn)
    test_data = Data(test_data, test_groups_data, opt.sample_order, opt.si_dc)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, pin_memory=True, shuffle=False, num_workers=8, collate_fn=collate_fn)
    print('end data initialization', datetime.datetime.now())

    model = trans_to_cuda(MLEUP(opt, n_node))
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    if opt.is_train == True:
        start_epoch = 0
    else:
        checkpoint = torch.load(opt.datasets + "/result_best.pth")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('load model success')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
    scheduler.last_epoch = start_epoch

    top_K = [1, 5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]
    if opt.is_train == True:
        for epoch in range(start_epoch, opt.epoch):
            print('-------------------------------------------------------')
            print('epoch: ', epoch)
            print('start training: ', datetime.datetime.now())
            model.train()
            total_loss = 0.0
            for data in tqdm(train_loader):
                optimizer.zero_grad()
                targets, scores_p, scores_d = forward(model, data)
                loss_p = loss_function(scores_p, targets)
                loss_d = loss_function(scores_d, targets)
                loss = loss_p + opt.β * (loss_d)
                loss.backward()
                optimizer.step()
                total_loss += loss
            scheduler.step()
            print('\tLoss:\t%.3f' % total_loss)
            print('end training: ', datetime.datetime.now())

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, opt.datasets + '/result_best_%s.pth' % (str(epoch)))

            print('start predicting: ', datetime.datetime.now())
            metrics = {}
            for K in top_K:
                metrics['hit%d' % K] = []
                metrics['mrr%d' % K] = []
                metrics['ndcg%d' % K] = []
            model.eval()
            for data in tqdm(test_loader):
                targets, scores_p, scores_d= forward(model, data)
                scores_p = trans_to_cpu(scores_p).detach().numpy()
                index = np.argsort(-scores_p, 1)
                targets = trans_to_cpu(targets).detach().numpy()
                for K in top_K:
                    for predict_item, target_item in zip(index[:, :K], targets):
                        metrics['hit%d' % K].append(np.isin(target_item, predict_item))
                        if len(np.where(predict_item == target_item)[0]) == 0:
                            metrics['mrr%d' % K].append(0)
                            metrics['ndcg%d' % K].append(0)
                        else:
                            metrics['mrr%d' % K].append(1 / (np.where(predict_item == target_item)[0][0] + 1))
                            metrics['ndcg%d' % K].append(1 / (np.log2(np.where(predict_item == target_item)[0][0] + 2)))

            for K in top_K:
                metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
                metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
                metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K]) * 100
                if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                    best_results['metric%d' % K][0] = metrics['hit%d' % K]
                    best_results['epoch%d' % K][0] = epoch
                if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                    best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                    best_results['epoch%d' % K][1] = epoch
                if best_results['metric%d' % K][2] < metrics['ndcg%d' % K]:
                    best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
                    best_results['epoch%d' % K][2] = epoch
            print(metrics)
            print('P@01\tP@05\tM@05\tN@05\tP@10\tM@10\tN@10\tP@20\tM@20\tN@20\t')
            print("%.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f" % (
                best_results['metric1'][0], best_results['metric5'][0], best_results['metric5'][1],
                best_results['metric5'][2], best_results['metric10'][0], best_results['metric10'][1],
                best_results['metric10'][2], best_results['metric20'][0], best_results['metric20'][1],
                best_results['metric20'][2]))
            print("%d\t\t %d\t\t %d\t\t %d\t\t %d\t\t %d\t\t %d\t\t %d\t\t %d\t\t %d" % (
                best_results['epoch1'][0], best_results['epoch5'][0], best_results['epoch5'][1],
                best_results['epoch5'][2], best_results['epoch10'][0], best_results['epoch10'][1],
                best_results['epoch10'][2], best_results['epoch20'][0], best_results['epoch20'][1],
                best_results['epoch20'][2]))
    else:
        print('start predicting: ', datetime.datetime.now())
        metrics = {}
        for K in top_K:
            metrics['hit%d' % K] = []
            metrics['mrr%d' % K] = []
            metrics['ndcg%d' % K] = []
        model.eval()
        for data in test_loader:
            targets, scores_p, scores_d = forward(model, data)
            scores_p = trans_to_cpu(scores_p).detach().numpy()
            index = np.argsort(-scores_p, 1)
            targets = trans_to_cpu(targets).detach().numpy()
            for K in top_K:
                for predict_item, target_item in zip(index[:, :K], targets):
                    metrics['hit%d' % K].append(np.isin(target_item, predict_item))
                    if len(np.where(predict_item == target_item)[0]) == 0:
                        metrics['mrr%d' % K].append(0)
                        metrics['ndcg%d' % K].append(0)
                    else:
                        metrics['mrr%d' % K].append(1 / (np.where(predict_item == target_item)[0][0] + 1))
                        metrics['ndcg%d' % K].append(1 / (np.log2(np.where(predict_item == target_item)[0][0] + 2)))

        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
            if best_results['metric%d' % K][2] < metrics['ndcg%d' % K]:
                best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
        print(metrics)
        print('P@01\tP@05\tM@05\tN@05\tP@10\tM@10\tN@10\tP@20\tM@20\tN@20\t')
        print("%.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f" % (
            best_results['metric1'][0], best_results['metric5'][0], best_results['metric5'][1],
            best_results['metric5'][2], best_results['metric10'][0], best_results['metric10'][1],
            best_results['metric10'][2], best_results['metric20'][0], best_results['metric20'][1],
            best_results['metric20'][2]))


if __name__ == '__main__':
    main()
