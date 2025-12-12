import torch
import numpy as np
import options
from datasets import RGDataset, FisVDataset, FS1000Dataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import model, loss
import os
from torch import nn
import train
from test import test_epoch
import math


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def get_optim(model, args):
    if args.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'rmsprop':
        optim = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise Exception("Unknown optimizer")
    return optim


def get_scheduler(optim, args):
    if args.lr_decay is not None:
        if args.lr_decay == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=args.epoch - args.warmup, eta_min=args.lr * args.decay_rate)
        elif args.lr_decay == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[args.epoch - 30], gamma=args.decay_rate)
        else:
            raise Exception("Unknown Scheduler")
    else:
        scheduler = None
    return scheduler


# Compute and print average metrics
def compute_average(metric_list):
    """
    metric_list: list of tuples, each tuple is (rho, mse)
    For metric 1, use Fisher Z transformation to compute mean then inverse transform:
        z = 0.5 * (log(1 + r) - log(1 - r))
        z_mean = mean(z)
        r_avg = (exp(z_mean) - exp(-z_mean))/(exp(z_mean) + exp(-z_mean))
    For metric 2, compute arithmetic mean.
    """
    r_values = [x[0] for x in metric_list]
    z_list = []
    for r in r_values:
        # Ensure r is in (-1, 1)
        z_list.append(0.5 * (math.log(1 + r) - math.log(1 - r)))
    zz = np.mean(z_list)
    num1_avg = (np.exp(zz) - np.exp(-zz)) / (np.exp(zz) + np.exp(-zz))
    num2_avg = np.mean([x[1] for x in metric_list])
    return num1_avg, num2_avg
        

if __name__ == '__main__':
    args = options.parser.parse_args()
    setup_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    '''
    1. load data
    '''
    '''
    train data
    '''
    Dataset = FS1000Dataset
    if args.dataset == 'RG':
        Dataset = RGDataset
    elif args.dataset == 'FisV':
        Dataset = FisVDataset
    elif args.dataset == 'FS1000':
        Dataset = FS1000Dataset
    train_data = Dataset(args.video_path, args.audio_path, args.flow_path, args.train_label_path,
                           clip_num=args.clip_num,
                           action_type=args.action_type, args=args)
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=8)
    print(len(train_data))
    # print(train_data.get_score_mean(), train_data.get_score_std())
    # raise SystemExit

    '''
    test data
    '''
    test_data = Dataset(args.video_path, args.audio_path, args.flow_path, args.test_label_path,
                          clip_num=args.clip_num,
                          action_type=args.action_type, train=False, args=args)
    test_loader = DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=8)
    print('=============Load dataset successfully=============')

    '''
    2. load model
    '''
    model = model.MoE_AQA(args.in_dim, args.hidden_dim, args.n_head, args.n_encoder,
                          args.n_decoder, args.n_query, args.dropout, args).to(device)
    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")
    print(f"Total learnable items: {len(enabled)}")
    loss_fn = loss.LossFun(args.alpha, args.margin)
    train_fn = train.train_epoch
    if args.ckpt is not None:
        checkpoint = torch.load('./ckpt/' + args.ckpt + '.pkl')
        model.load_state_dict(checkpoint)
    print('=============Load model successfully=============')

    print(args)

    '''
    test mode
    '''
    if args.test:
        combinations = [
            [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ]  # Triplet corresponds to [V, A, F]
        results = []  # Store test results for each combination
        results_for_avg = []  # Store results excluding [1,1,1] for averaging

        # Define modality names
        modalities = ['V', 'A', 'F']

        for i, mask in enumerate(combinations):
            # Generate modality combination name
            combination_name = ''.join([modalities[j] for j in range(3) if mask[j] == 1])
            # Get description of current combination
            vaf_desc = f"{combination_name}"
            test_loss, coef = test_epoch(0, model, test_loader, None, device, mask, args)
            # Store results in list
            results.append({
                "combination": mask,
                "name": combination_name,
                "test_loss": test_loss,
                "test_coef": coef
            })
            # Store results for averaging (exclude [1,1,1])
            if mask != [1, 1, 1]:
                results_for_avg.append((coef, test_loss))
            # Print test results for current combination
            print(f"Combination {i + 1}: {combination_name}")
            print('Test Loss: {:.4f}\tTest Coef: {:.3f}'.format(test_loss, coef))

        # Print summary of test results for all combinations
        print("\nTest results summary:")
        for result in results:
            print(f"Combination: {result['name']}\t"
                  f"Test Loss: {result['test_loss']:.2f}\t"
                  f"Test Coef: {result['test_coef']:.3f}")

        avg_coef, avg_loss = compute_average(results_for_avg)
        print(f"\nAverage metrics (excluding VAF):")
        print(f"Average Test Loss: {avg_loss:.4f}\tAverage Test Coef: {avg_coef:.3f}")

        raise SystemExit

    '''
    3. record
    '''
    if not os.path.exists("./ckpt/"):
        os.makedirs("./ckpt/")
    if not os.path.exists("./logs/" + args.model_name):
        os.makedirs("./logs/" + args.model_name)
    logger = SummaryWriter(os.path.join('./logs/', args.model_name))
    best_coef, best_epoch, best_mse, best_epoch2 = -1, -1, 100000., -1
    final_train_loss, final_train_coef, final_test_loss, final_test_coef = 0, 0, 0, 0

    '''
    4. train
    '''
    optim = get_optim(model, args)
    scheduler = get_scheduler(optim, args)
    print('=============Begin training=============')
    if args.warmup:
        warmup = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda t: t / args.warmup)
    else:
        warmup = None

    for epc in range(args.epoch):
        if args.warmup and epc < args.warmup:
            warmup.step()
        # print(optim.state_dict()['param_groups'][0]['lr'])
        avg_loss, train_coef = train_fn(epc, model, loss_fn, train_loader, optim, logger, device, args)
        if scheduler is not None and (args.lr_decay != 'cos' or epc >= args.warmup):
            scheduler.step()
        test_loss, test_coef = test_epoch(epc, model, test_loader, logger, device, [1, 1, 1], args)
        if test_coef > best_coef:
            best_coef, best_epoch = test_coef, epc
            torch.save(model.state_dict(), './ckpt/' + args.model_name + '_best.pkl')
        if test_loss < best_mse:
            best_mse, best_epoch2 = test_loss, epc
            torch.save(model.state_dict(), './ckpt/' + args.model_name + '_best_mse.pkl')
        print('Epoch: {}\tLoss: {:.4f}\tTrain Coef: {:.3f}\tTest Loss: {:.4f}\tTest Coef: {:.3f}'
              .format(epc, avg_loss, train_coef, test_loss, test_coef))
        if epc == args.epoch - 1:
            final_train_loss, final_train_coef, final_test_loss, final_test_coef = \
                avg_loss, train_coef, test_loss, test_coef
    # torch.save(model.state_dict(), './ckpt/' + args.model_name + '.pkl')
    print('Best Test Coef: {:.3f}\tBest Test Eopch: {}\tBest Test Mse: {:.3f}\tBest Mse Eopch: {}'.format(best_coef, best_epoch, best_mse, best_epoch2))
