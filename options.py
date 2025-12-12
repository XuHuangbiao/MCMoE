import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--video-path', type=str, default='./dataset/FS1000/output_feature_fs1000_new')
parser.add_argument('--audio-path', type=str, default='./dataset/FS1000/ast_feature_fs1000_new')
parser.add_argument('--flow-path', type=str, default='./dataset/FS1000/i3d_avg_clip8_5s_fs1000')
parser.add_argument('--clip-num', type=int, default=95)

parser.add_argument('--dataset', type = str, choices=['FS1000', 'FisV', 'RG'], default='FS1000')
parser.add_argument('--train-label-path', type=str, default='./dataset/FS1000/train_fs1000_new.txt')
parser.add_argument('--test-label-path', type=str, default='./dataset/FS1000/val_fs1000_new.txt')

parser.add_argument('--action-type', type=str, default='TES')
parser.add_argument('--score-type', type=str, default='Total_Score')

parser.add_argument('--model-name', type=str, default='action_net', help='name used to save model and logs')
parser.add_argument("--ckpt", default=None, help="ckpt for pretrained model")
parser.add_argument("--test", action='store_true', help="only evaluate, don't train")

parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--optim', type=str, default='adam')

parser.add_argument("--lr-decay", type=str, default=None, help='use what decay scheduler')
parser.add_argument("--decay-rate", type=float, default=0.1, help="lr decay rate")
parser.add_argument("--warmup", type=int, default=0, help="warmup epoch")

parser.add_argument('--in_dim', type=int, default=768)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_encoder', type=int, default=3)
parser.add_argument('--n_decoder', type=int, default=3)
parser.add_argument('--n_query', type=int, default=4)
parser.add_argument("--use_pe", type=bool, default=False)

parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--margin', type=float, default=1.0)

parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--score_range', type=int, default=None)
