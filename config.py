import  argparse

args = argparse.ArgumentParser()
# args.add_argument('--dataset', default='cora')
args.add_argument('--learning_rate', type=float, default=0.01)
args.add_argument('--epochs', type=int, default=200)
args.add_argument('--attack_epochs', type=int, default=20)
args.add_argument('--attack_lr', type=float, default=.1)
args.add_argument('--init_z', type=str, default='randn')
args.add_argument('--sigma', type=float, default=1.)
args.add_argument('--mu', type=float, default=0.)
args.add_argument('--seed', type=int, default=49)
args.add_argument('--dev', type=str, default='cpu')
args.add_argument('--size', type=int, default=100)
args.add_argument('--kk', type=int, default=0)
args.add_argument('--hidden', type=int, default=40)
args.add_argument('--eye', type=float, default=1.)
args.add_argument('--dropout', type=float, default=0.)
args.add_argument('--lr_decay', type=float, default=0.95)
# args.add_argument('--max_degree', type=int, default=3)


args = args.parse_args()
print(args)
