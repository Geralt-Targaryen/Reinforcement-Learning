import argparse
from DDPG import DDPG, Critic, Actor
from DDQN import DDQN, DQNet

parser = argparse.ArgumentParser()
parser.add_argument('--env', required=True, help='name of environment, eg. Ant-v2, BoxingNoFrameskip-v4', type=str)
# network hyperparameters
parser.add_argument('--lr_start', default=5e-5, help='initial learning rate', type=float)
parser.add_argument('--lr_end', default=1e-6, help='learning rate in the end', type=float)
parser.add_argument('--h', default=256, help='size of networks\' hidden layers', type=int)
parser.add_argument('--weight_decay', default=1e-4, help='weight decay factor', type=float)
parser.add_argument('--batch_size', default=128, help='batch size sampled from the buffer', type=int)
# RL hyperparameters
parser.add_argument('--type', choices=['policy', 'value'], default='value', help='policy based or value based', type=str)
parser.add_argument('--dueling', action='store_true', help='whether to use dueling architecture')
parser.add_argument('--soft_update', action='store_true', help='whether to use soft update')
parser.add_argument('--buffer_size', default=4096, help='buffer size', type=int)
parser.add_argument('--eps_start', default=0.2, help='initial exploration noise', type=float)
parser.add_argument('--eps_end', default=0.05, help='exploration noise in the end', type=float)
parser.add_argument('--gamma', default=0.99, help='discount factor', type=float)
parser.add_argument('--total_steps', default=10000000, help='total training time steps', type=int)
parser.add_argument('--update_steps', default=4000, help='interval for network hard update', type=int)
parser.add_argument('--tau', default=0.05, help='factor for network soft update', type=float)
# auxiliary hyperparameters
parser.add_argument('--train', action='store_true', help='train a new agent')
parser.add_argument('--eval', action='store_true', help='evaluate an existing agent')
parser.add_argument('--save_steps', default=500000, help='interval for saving checkpoints', type=int)
parser.add_argument('--model', default=None, help='model (actor) to be evaluated', type=str)

args = parser.parse_args()


def run():
    if args.eval:
        if not args.model:
            print('Please specified a model for evaluation, use python run.py --help for more info')
            return
        if args.type == 'policy':
            ddpg = DDPG(args.env)
            ddpg.eval(args.model)
        else:
            ddqn = DDQN(args.env)
            ddqn.eval(args.model)

    elif args.train:
        if args.type == 'policy':
            ddpg = DDPG(args.env, args.lr_start, args.lr_end, args.batch_size, args.eps_start, args.eps_end,
                        args.gamma, args.weight_decay, args.total_steps, args.save_steps, args.h,
                        args.buffer_size, args.dueling, args.soft_update, args.update_steps, args.tau)
            ddpg.train()
        else:
            ddqn = DDQN(args.env, args.lr_start, args.lr_end, args.batch_size, args.eps_start, args.eps_end,
                        args.gamma, args.weight_decay, args.total_steps, args.save_steps, args.h,
                        args.buffer_size, args.dueling, args.soft_update, args.update_steps, args.tau)
            ddqn.train()


if __name__ == '__main__':
    run()







