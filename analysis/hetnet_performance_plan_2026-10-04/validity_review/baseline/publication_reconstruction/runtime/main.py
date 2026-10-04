# Resolve the frozen environment copy in the parent and spawned collectors.
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'envs'))

import argparse
import atexit
import signal
import sys
import time
RUNTIME_BEGIN = time.monotonic()
import signal
import argparse
import os
import hashlib
import json
import resource

# Library pools must be constrained before NumPy/Torch/DGL import.
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import tracemalloc

import numpy as np
import torch
import visdom

# Deviation A: one CPU thread per collector, matching the four-core job contract.
torch.set_num_threads(1)

import data
from action_utils import parse_action_args
from comm import CommNetMLP
from hetgat.policy import A2CPolicy, PGPolicy
from models import *
from multi_processing import MultiProcessTrainer
from trainer import Trainer
from eval_trainer import EvalTrainer
from utils import *
from pathlib import Path
from hetnet_ext.seeding import seed_everything
from hetnet_ext.recording import TrainingRecorder, write_checkpoint_signature, record_checkpoint
from hetnet_ext.recording import write_json_new
from hetnet_ext.recovery import (RECOVERY_VERSION, atomic_checkpoint, capture_rng,
    restore_rng, rng_scheme, scientific_arguments, seed_stream, validate_recovery,
    process_resources)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='Isolated historical HetNet reconstruction')
parser.add_argument('--publication_env_version', required=True,
                    choices=['historical-2022', 'corrected-v1'])
parser.add_argument('--max_env_steps', type=int, default=0,
                    help='stop after a complete update reaching this total; zero disables')
parser.add_argument('--source_manifest', required=True)
parser.add_argument('--model_spec', choices=['public-code-v1', 'supplement-v1'], default='public-code-v1')
parser.add_argument('--milestones', nargs='*', type=int, default=[])
parser.add_argument('--resume_checkpoint', default='')
parser.add_argument('--wall_seconds', type=float, default=0,
                    help='pause after a complete update once this runtime duration is reached; zero disables')
parser.add_argument('--episode_log', choices=['file', 'stdout'], default='file')
# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses

parser.add_argument('--experiment_name', default='experiment', type=str,
                    help='name of the experiment')
parser.add_argument('--save_dir', default='./saved', type=str, help='directory to save models')
parser.add_argument('--metrics_file', default='', type=str,
                    help='append epoch JSONL and signatures to a fresh run directory')
parser.add_argument('--profile_memory', action='store_true', default=False,
                    help='trace Python allocations for memory diagnostics (slows training)')


parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=10,
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=16,
                    help='How many processes to run')
# model
parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
parser.add_argument('--recurrent', action='store_true', default=False,
                    help='make the model recurrent in time')

parser.add_argument('--use_binary', default=False, action='store_true',
                    help='Wheather to use binarization in hetgat')
parser.add_argument('--msg_dim', default=16, type=int,
                    help='Message size of binarization')

# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--tau', type=float, default=1.0,
                    help='gae (remove?)')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed. Pass -1 for random seed')  # TODO: works in thread?
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')
# environment
parser.add_argument('--env_name', default="Cartpole",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')
parser.add_argument('--comm_range_P', default=-1, type=int,
                    help='range perception agents can communicate in (-1 for infinite)')
parser.add_argument('--comm_range_A', default=-1, type=int,
                    help='range action agents can communicate in (-1 for infinite)')
parser.add_argument('--lossy_comm', action='store_true', default=False,
                    help='communication is lost as range approaches maximum')
parser.add_argument('--min_comm_loss', default=0.0, type=float,
                    help='percentage of communication lost at no range')
parser.add_argument('--max_comm_loss', default=0.3, type=float,
                    help='percentage of communication lost at max range')

# other
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
parser.add_argument('--save', default='tst', type=str,
                    help='save the model after training')
parser.add_argument('--save_every', default=10, type=int,
                    help='save the model after every n_th epoch')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--display', action="store_true", default=False,
                    help='Display environment state')
parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")
parser.add_argument('--use_cuda', action='store_true', default=False,
                    help='use cuda instead of cpu')

# hetgat specific args
parser.add_argument('--hetgat', action='store_true', default=False,
                    help="enable hetgat model")
parser.add_argument('--lr_gamma', type=float, default=0.1,
                    help="lr gamma parameter")
parser.add_argument('--hetgat_a2c', action='store_true', default=False,
                    help="enable hetgat a2c model")
parser.add_argument('--eval', action='store_true', default=False,
                    help='evaluate a model')
parser.add_argument('--eval_string', default='', type=str,
                    help='string that will be used to save result')
parser.add_argument('--eval_config', default='', type=str,
                    help='holds all of the evaluation starting conditions')


# CommNet specific args
parser.add_argument('--commnet', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--hetcomm', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--ic3net', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--nagents', type=int, default=1,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--comm_mode', type=str, default='avg',
                    help="Type of mode for communication tensor calculation [avg|sum]")
parser.add_argument('--comm_passes', type=int, default=1,
                    help="Number of comm passes per step over the model")
parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                    help="Whether communication should be there")
parser.add_argument('--mean_ratio', default=1.0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--rnn_type', default='MLP', type=str,
                    help='type of rnn to use. [LSTM|MLP]')
parser.add_argument('--detach_gap', default=10000, type=int,
                    help='detach hidden state and cell state for rnns at this interval.'
                         + ' Default 10000 (very high)')
parser.add_argument('--total_state_action_in_batch', default=500, type=int,
                    help='number of s,a in a batch.'
                         + ' Default 500 (very high)')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--hard_attn', default=False, action='store_true',
                    help='Whether to use hard attention: action - talk|silent')
parser.add_argument('--comm_action_one', default=False, action='store_true',
                    help='Whether to always talk, sanity check for hard attention.')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='Whether to multipy log prob for each chosen action with advantages')
parser.add_argument('--share_weights', default=False, action='store_true',
                    help='Share weights for hops')

# Deviation B: even the environment used to register CLI arguments is constructed
# after seeding. Worker run() subsequently retains its seed + id + 1 stream.
resolved_seed = seed_everything(parser.parse_known_args()[0].seed)
if parser.parse_known_args()[0].model_spec == 'supplement-v1':
    seed_stream(resolved_seed)
init_args_for_env(parser)
args = parser.parse_args()
args.seed = resolved_seed
args.rng_scheme = rng_scheme(args.model_spec)
if args.wall_seconds < 0 or not np.isfinite(args.wall_seconds):
    raise ValueError('wall_seconds must be finite and nonnegative')
if any(value <= 0 for value in args.milestones) or args.milestones != sorted(set(args.milestones)):
    raise ValueError('milestones must be positive, strictly increasing step thresholds')
if args.resume_checkpoint and args.load:
    raise ValueError('Use resume_checkpoint without legacy load')
if args.model_spec == 'supplement-v1' and (not args.hetgat_a2c or args.use_cuda):
    raise ValueError('supplement-v1 requires the CPU HetGAT A2C runtime')

if args.comm_range_P == -1 or args.comm_range_A == -1:
    args.lossy_comm = False

if args.ic3net:
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0

    # For TJ set comm action to 1 as specified in paper to showcase
    # importance of individual rewards even in cooperative games
    if args.env_name == "traffic_junction":
        args.comm_action_one = True

if not hasattr(args, 'nfriendly_P') and not hasattr(args, 'nfriendly_A'):
    args.nfriendly_A = 1
    args.nfriendly_P = args.nagents - args.nfriendly_A

args.nfriendly = args.nfriendly_P + args.nfriendly_A

# Enemy comm
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

env = data.init(args.env_name, args, False)

num_inputs = env.observation_dim
args.num_actions = env.num_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)):  # single action case
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
args.num_inputs = num_inputs

# Hard attention
if args.hard_attn and args.commnet:
    # add comm_action as last dim in actions
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = env.dim_actions + 1

# Recurrence
if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'
if args.hetcomm:
    args.recurrent = True
    args.rnn_type = 'LSTM'

parse_action_args(args)

print(args)

if args.commnet:
    policy_net = CommNetMLP(args, num_inputs, 4)
    print(policy_net)
elif args.hetgat:
    pos_len = args.dim ** 2
    SSN_state_len = 4

    in_dim_raw = {'vision': args.vision,
                  'P': pos_len + SSN_state_len,
                  'A': pos_len,
                  'state': SSN_state_len
                  }
    in_dim = {'P': pos_len + SSN_state_len,
              'A': pos_len,
              'state': SSN_state_len}
    hid_dim = {'P': 16,
               'A': 16,
               'state': 16}
    out_dim = {'P': 5,
               'A': 6,
               'state': 8}
    with_two_state = True
    # if with_two_state:
    #     in_dim['state'] = SSN_state_len
    num_heads = 4

    device_name = 'cuda' if args.use_cuda else 'cpu'
    device = torch.device(device_name)

    obs = None if not hasattr(args, 'vision') else (2 * args.vision + 1) ** 2
    tensor_obs = None if not hasattr(args, 'tensor_obs') else args.tensor_obs

    milestones = [200, 400]
    if args.hetgat_a2c:
        policy = A2CPolicy(in_dim_raw, in_dim, hid_dim, out_dim, args.nfriendly_P,
                           args.nfriendly_A, num_heads=num_heads, msg_dim=args.msg_dim,
                           device=device, gamma=args.gamma, lr=args.lrate, weight_decay=0,
                           milestones=milestones, lr_gamma=0.1, use_real=(not args.use_binary),
                           use_CNN=False, use_tanh=False, per_class_critic=True,
                           per_agent_critic=False, with_two_state=with_two_state, obs=obs,
                           comm_range_P=args.comm_range_P, comm_range_A=args.comm_range_A,
                           model_spec=args.model_spec,
                           lossy_comm=args.lossy_comm, min_comm_loss=args.min_comm_loss,
                           max_comm_loss=args.max_comm_loss, tensor_obs=tensor_obs, action_vision=args.A_vision)
    else:
        policy = PGPolicy(in_dim_raw, in_dim, hid_dim, out_dim, num_heads=num_heads,
                          device=device, gamma=args.gamma, lr=args.lrate,
                          weight_decay=0, milestones=milestones, lr_gamma=0.1,
                          use_real=True, use_CNN=False)
    policy_net = policy.model
elif args.random:
    policy_net = Random(args, num_inputs)
elif args.recurrent:
    policy_net = RNN(args, num_inputs)
else:
    policy_net = MLP(args, num_inputs)

if args.hetcomm:
    pass
else:
    if not args.display:
        display_models([policy_net])

# share parameters among threads, but not gradients
if args.hetcomm:
    for p in policy_action_net.parameters():
        p.data.share_memory_()

    for p in policy_perception_net.parameters():
        p.data.share_memory_()

else:
    for p in policy_net.parameters():
        p.data.share_memory_()

if args.nprocesses > 1:
    if args.hetgat:
        if __name__ == '__main__':
            trainer = MultiProcessTrainer(args,
                                          lambda: Trainer(args, policy_net, data.init(args.env_name, args), policy))
    else:
        if __name__ == '__main__':
            trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, data.init(args.env_name, args)))
else:
    if args.hetgat:
        if args.eval:
            
            
            # trainer = EvalTrainer(args, policy_net, data.init(args.env_name, args), policy)
            trainer = Trainer(args, policy_net, data.init(args.env_name, args), policy)

            
            
        else:
            trainer = Trainer(args, policy_net, data.init(args.env_name, args), policy)
    elif args.hetcomm:
        trainer = Trainer(args, [policy_perception_net, policy_action_net], data.init(args.env_name, args))
    else:
        if args.eval:
            trainer = EvalTrainer(args, policy_net, data.init(args.env_name, args))
        else:
            trainer = Trainer(args, policy_net, data.init(args.env_name, args))
if __name__ == '__main__' and args.nprocesses > 1:
    # Covers failures in later initialization as well as the training finally.
    atexit.register(trainer.quit)
if args.hetcomm:
    disp_trainer = Trainer(args, [policy_perception_net, policy_action_net], data.init(args.env_name, args, False))
else:
    disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))

disp_trainer.display = True


def disp():
    x = disp_trainer.get_episode()


log = dict()
log['epoch'] = LogField(list(), False, None, None)
log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')
log['enemy_count'] = LogField(list(), True, 'epoch', 'num_steps')

log['num_episodes'] = LogField(list(), True, 'epoch', None)
log['num_steps'] = LogField(list(), True, 'epoch', None)

if args.plot:
    vis = visdom.Visdom(env=args.plot_env)

model_dir = Path(args.save_dir) / args.experiment_name

if not model_dir.exists():
    curr_run = 'run1'
else:
    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                     model_dir.iterdir() if
                     str(folder.name).startswith('run')]
    if len(exst_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums) + 1)
run_dir = model_dir / curr_run
progress = dict(env_steps=0, episodes=0, updates=0, epoch=0)
run_state = dict(completed_epochs=0, updates_in_epoch=0, epoch_stat={},
                 epoch_elapsed_seconds=0.0, active_time_seconds=0.0, milestones_reached=[])
recorder = None
source_sha256 = hashlib.sha256(Path(args.source_manifest).read_bytes()).hexdigest()
soft_stop_requested = False
checkpoint_seconds = 0.0
segment_active_base = 0.0
segment_begin = None


def all_rng_states():
    return trainer.rng_states() if args.nprocesses > 1 else [capture_rng()]


def restore_all_rng_states(states):
    if args.nprocesses > 1:
        trainer.restore_rng_states(states)
    else:
        if len(states) != 1:
            raise ValueError('Recovery collector count differs')
        restore_rng(states[0])


def save_checkpoint(filename, reason):
    global checkpoint_seconds
    recovery = dict(run_state)
    recovery.update(version=RECOVERY_VERSION, counts=dict(progress),
                    recorder_state=recorder.state_dict() if recorder else None,
                    rng_states=all_rng_states(), scientific_args=scientific_arguments(args),
                    stop_reason=reason)
    # The immutable segment base prevents repeated snapshots from counting the
    # same elapsed interval twice.  Includes logging/checkpoints before this
    # snapshot, but excludes process startup and downtime between segments.
    recovery['active_time_seconds'] = segment_active_base + time.monotonic() - segment_begin
    d = dict(policy_net=policy_net.state_dict(), log=log,
             trainer=trainer.state_dict(), seed=args.seed, recovery=recovery)
    d['reconstruction'] = dict(schema_version=2, env_version=args.publication_env_version,
        model_spec=args.model_spec, rng_scheme=args.rng_scheme,
        resolved_args=vars(args), counts=dict(progress), source_manifest_sha256=source_sha256)
    checkpoint = run_dir / ('model_update%08i_%s' % (progress['updates'], filename.removeprefix('model_')))
    checkpoint_begin_time = time.monotonic()
    atomic_checkpoint(checkpoint, d)
    if args.metrics_file:
        signature = write_checkpoint_signature(checkpoint, policy_net)
        record_checkpoint(args.metrics_file, checkpoint, progress['epoch'],
                          time.monotonic() - checkpoint_begin_time, signature, counts=progress)
    checkpoint_seconds += time.monotonic() - checkpoint_begin_time
    return checkpoint


def finish_epoch(epoch, cpu_mem_peak, gpu_mem_peak):
    """Close one full epoch, or the final partial scientific-budget epoch."""
    stat = dict(run_state['epoch_stat'])
    epoch_time = run_state['epoch_elapsed_seconds']
    if recorder is not None:
        metrics = recorder.finish_epoch(epoch, epoch_time, policy_net,
            updates=progress['updates'], updates_in_epoch=run_state['updates_in_epoch'])
        print(json.dumps({'record_type': 'publication_epoch', **metrics},
                         sort_keys=True, allow_nan=False), flush=True)
    for key, field in log.items():
        if key == 'epoch':
            field.data.append(epoch)
        elif key == 'enemy_count':
            field.data.append(stat.get(key, []))
        else:
            if key in stat and field.divide_by is not None and stat[field.divide_by] > 0:
                stat[key] = stat[key] / stat[field.divide_by]
            field.data.append(stat.get(key, 0))
    np.set_printoptions(precision=2)
    cpu_memory = '{}MB'.format(cpu_mem_peak / 10 ** 6) if args.profile_memory else 'disabled'
    print('Epoch {}\tReward {}\tTime {:.2f}s, Episodes {}, Total Steps {}, Python Allocation Peak {}, GPU Memory Peak {}MB'.format(
        epoch, stat['reward'], epoch_time, progress['episodes'], progress['env_steps'],
        cpu_memory, gpu_mem_peak / 10 ** 6))
    for key, label in [('enemy_reward', 'Enemy-Reward'), ('add_rate', 'Add-Rate'),
                       ('success', 'Success'), ('steps_taken', 'Steps-taken'),
                       ('comm_action', 'Comm-Action'), ('enemy_comm', 'Enemy-Comm')]:
        if key in stat:
            print('{}: {}'.format(label, stat[key]))
    if 'enemy_count' in stat:
        print('Average-Enemy-Count: {}'.format(np.average(stat['enemy_count'])))
    if args.plot:
        for key, field in log.items():
            if field.plot and field.data:
                vis.line(np.asarray(field.data), np.asarray(log[field.x_axis].data[-len(field.data):]),
                         win=key, opts=dict(xlabel=field.x_axis, ylabel=key))


def run(num_epochs):
    global recorder, segment_active_base, segment_begin
    resumed = None
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location='cpu')
        resumed = validate_recovery(checkpoint, args, source_sha256)
        policy_net.load_state_dict(checkpoint['policy_net'], strict=True)
        trainer.load_state_dict(checkpoint['trainer'])
        log.update(checkpoint['log'])
        progress.update(resumed['counts'])
        for name in run_state:
            run_state[name] = resumed[name]
    recorder = TrainingRecorder(args, policy_net) if args.metrics_file else None
    if resumed:
        if recorder is None or resumed['recorder_state'] is None:
            raise ValueError('Recovery requires the structured training recorder')
        recorder.load_state_dict(resumed['recorder_state'])
        restore_all_rng_states(resumed['rng_states'])
    elif args.model_spec == 'supplement-v1':
        # Initialization has a distinct stream; the parent is collector zero.
        seed_stream(args.seed, 0)
    if args.save:
        run_dir.mkdir(parents=True, exist_ok=True)
    if recorder:
        write_json_new(recorder.path.parent / 'training_segment.json', {
            'schema_version': 1, 'resume_checkpoint': args.resume_checkpoint or None,
            'resume_checkpoint_sha256': hashlib.sha256(Path(args.resume_checkpoint).read_bytes()).hexdigest()
                if args.resume_checkpoint else None,
            'starting_counts': dict(progress), 'model_spec': args.model_spec,
            'starting_active_time_seconds': run_state['active_time_seconds'],
            'rng_scheme': args.rng_scheme, 'source_manifest_sha256': source_sha256})

    run_begin = time.monotonic()
    segment_begin = run_begin
    segment_active_base = run_state['active_time_seconds']
    startup_seconds = run_begin - RUNTIME_BEGIN
    update_work_seconds = 0.0
    epoch_cpu_peak = np.zeros(args.nprocesses)
    epoch_gpu_peak = np.zeros(args.nprocesses)
    last_checkpoint = None
    reason = 'epoch_cap_completed'
    while run_state['completed_epochs'] < num_epochs:
        ep = run_state['completed_epochs']
        n = run_state['updates_in_epoch']
        print('[Epoch] batch', n)
        trainer.display = bool(n == args.epoch_size - 1 and args.display)
        update_begin = time.monotonic()
        fresh, cpu_peak, gpu_peak = trainer.train_batch(ep)
        update_seconds = time.monotonic() - update_begin
        update_work_seconds += update_seconds
        episodes = fresh.pop('_episode_records')
        trainer.display = False
        progress.update(env_steps=progress['env_steps'] + int(fresh['num_steps']),
                        episodes=progress['episodes'] + int(fresh['num_episodes']),
                        updates=progress['updates'] + 1, epoch=ep + 1)
        run_state['updates_in_epoch'] += 1
        merge_stat(fresh, run_state['epoch_stat'])
        if recorder:
            recorder.add_batch(fresh)
            recorder.record_update(fresh, episodes, progress, ep + 1,
                                   run_state['updates_in_epoch'], update_seconds)
        run_state['epoch_elapsed_seconds'] += time.monotonic() - update_begin
        epoch_cpu_peak = np.maximum(epoch_cpu_peak, cpu_peak)
        epoch_gpu_peak = np.maximum(epoch_gpu_peak, gpu_peak)
        budget_reached = bool(args.max_env_steps and progress['env_steps'] >= args.max_env_steps)
        full_epoch = run_state['updates_in_epoch'] == args.epoch_size
        if full_epoch or budget_reached:
            finish_epoch(ep + 1, epoch_cpu_peak, epoch_gpu_peak)
        if full_epoch:
            run_state.update(completed_epochs=ep + 1, updates_in_epoch=0,
                             epoch_stat={}, epoch_elapsed_seconds=0.0)
            epoch_cpu_peak[:] = 0
            epoch_gpu_peak[:] = 0

        # Every checkpoint observes the same completed optimizer update.  A
        # milestone inside an epoch retains that epoch's sums for continuation.
        crossed = [threshold for threshold in args.milestones
                   if threshold <= progress['env_steps'] and threshold not in run_state['milestones_reached']]
        # If one update crosses several thresholds, every snapshot must know
        # about all of them.  The first saved snapshot serves every such target
        # even if interruption prevents writing an additional duplicate file.
        run_state['milestones_reached'].extend(crossed)
        for threshold in crossed:
            if args.save:
                last_checkpoint = save_checkpoint('model_steps%i.pt' % threshold,
                    'budget_completed' if budget_reached else 'milestone')
        if budget_reached:
            reason = 'budget_completed'
            break
        if full_epoch and run_state['completed_epochs'] == num_epochs:
            reason = 'epoch_cap_completed'
            break
        if soft_stop_requested or (args.wall_seconds and time.monotonic() - run_begin >= args.wall_seconds):
            reason = 'paused_signal' if soft_stop_requested else 'paused_wall_time'
            break
        if full_epoch and args.save_every and (ep + 1) % args.save_every == 0 and args.save:
            last_checkpoint = save_checkpoint('model_ep%i.pt' % (ep + 1), 'periodic')

    if args.save:
        filename = ('model_paused.pt' if reason.startswith('paused_')
                    else 'model_ep%i.pt' % progress['epoch'])
        last_checkpoint = save_checkpoint(filename, reason)
    if recorder:
        segment_seconds = time.monotonic() - run_begin
        status = {'schema_version': 1, 'stop_reason': reason, 'counts': dict(progress),
                  'scientific_budget_completed': reason == 'budget_completed' or
                      (reason == 'epoch_cap_completed' and not args.max_env_steps),
                  'checkpoint': str(last_checkpoint.resolve()) if last_checkpoint else None,
                  'segment_wall_time_seconds': segment_seconds,
                  'active_time_seconds': segment_active_base + segment_seconds,
                  'startup_to_training_seconds': startup_seconds,
                  'training_update_seconds': update_work_seconds,
                  'checkpoint_seconds': checkpoint_seconds,
                  'resources': {
                      'collectors': trainer.collector_resources() if args.nprocesses > 1 else [process_resources()],
                      'completed_children_only': process_resources(resource.RUSAGE_CHILDREN),
                      'collector_count': args.nprocesses,
                      'slurm_cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK')}}
        write_json_new(recorder.path.parent / 'run_status.json', status)
        print(json.dumps({'record_type': 'publication_status', **status}, sort_keys=True), flush=True)
    return progress['epoch']


def load(path):
    d = torch.load(path)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])
    log.update(d['log'])
    trainer.load_state_dict(d['trainer'])


def signal_handler(signal, frame):
    print('Training interrupted by signal {}.'.format(signal), flush=True)
    if args.display:
        env.end_display()
    sys.exit(128 + signal)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def request_soft_stop(signum, frame):
    global soft_stop_requested
    soft_stop_requested = True


if hasattr(signal, 'SIGUSR1'):
    signal.signal(signal.SIGUSR1, request_soft_stop)

if __name__ == '__main__':
    try:
        if args.load != '':
            load(args.load)

        completed_epochs = run(args.num_epochs)

        if args.display:
            env.end_display()

    finally:
        if args.nprocesses > 1:
            trainer.quit()
