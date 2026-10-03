import time
import random
from utils import *
import torch
import torch.multiprocessing as mp
from hetnet_ext.recovery import capture_rng, restore_rng, seed_stream, process_resources


class MultiProcessWorker(mp.Process):
    # TODO: Make environment init threadsafe
    def __init__(self, id, trainer_maker, comm, seed, *args, **kwargs):
        self.id = id
        self.seed = seed
        super(MultiProcessWorker, self).__init__()
        self.trainer = trainer_maker()
        self.trainer.collector_id = id + 1
        self.comm = comm

    def run(self):
        if getattr(self.trainer.args, 'model_spec', 'public-code-v1') == 'supplement-v1':
            seed_stream(self.seed, self.id + 1)
        else:
            torch.manual_seed(self.seed + self.id + 1)
            np.random.seed(self.seed + self.id + 1)
            random.seed(self.seed + self.id + 1)

        while True:
            task = self.comm.recv()
            if type(task) == list:
                task, epoch = task

            if task == 'quit':
                return
            elif task == 'run_batch':
                batch, stat = self.trainer.run_batch(epoch)
                self.trainer.optimizer.zero_grad(set_to_none=False)
                s = self.trainer.compute_grad(batch)
                merge_stat(s, stat)

                self.comm.send(stat)
            elif task == 'send_grads':
                grads = []
                for p in self.trainer.params:
                    if p._grad is not None:
                        grads.append(p._grad.data)

                self.comm.send(grads)
            elif task == 'get_gpu_mem':
                self.comm.send((self.id, self.trainer.get_memory_peak()))
            elif task == 'reset_gpu_mem':
                self.trainer.reset_memory_peak()
            elif task == 'get_rng':
                self.comm.send(capture_rng())
            elif task == 'set_rng':
                restore_rng(epoch)
                self.comm.send(True)
            elif task == 'get_resources':
                self.comm.send(process_resources())


class MultiProcessTrainer(object):
    def __init__(self, args, trainer_maker):
        self.comms = []
        self.workers = []
        self.trainer = trainer_maker()
        # itself will do the same job as workers
        self.nworkers = args.nprocesses - 1

        try:
            for i in range(self.nworkers):
                comm, comm_remote = mp.Pipe()
                self.comms.append(comm)
                worker = MultiProcessWorker(i, trainer_maker, comm_remote, seed=args.seed)
                self.workers.append(worker)
                worker.start()
                comm_remote.close()
        except BaseException:
            self.quit()
            raise

        self.grads = None
        self.worker_grads = None
        self.is_random = args.random
        try:
            self.reset_mem_peak()
        except BaseException:
            self.quit()
            raise

    def quit(self):
        for comm in self.comms:
            try:
                comm.send('quit')
            except (BrokenPipeError, EOFError, OSError):
                pass
        # A failed or interrupted collector may never return to its command
        # loop. Bound shutdown, then terminate only this trainer's workers.
        deadline = time.monotonic() + 2
        for worker in self.workers:
            if worker.pid is not None:
                worker.join(timeout=max(0, deadline - time.monotonic()))
        for worker in self.workers:
            if worker.pid is not None and worker.is_alive():
                worker.terminate()
        deadline = time.monotonic() + 2
        for worker in self.workers:
            if worker.pid is not None:
                worker.join(timeout=max(0, deadline - time.monotonic()))
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=1)
        for comm in self.comms:
            comm.close()
        self.comms = []
        self.workers = []

    def obtain_grad_pointers(self):
        # only need perform this once
        if self.grads is None:
            self.grads = []
            for p in self.trainer.params:
                if p._grad is not None:
                    self.grads.append(p._grad.data)

        if self.worker_grads is None:
            self.worker_grads = []
            for comm in self.comms:
                comm.send('send_grads')
                grad = comm.recv()

                self.worker_grads.append(grad)

    def train_batch(self, epoch):
        self.cpu_memory_peak = np.zeros((self.nworkers + 1))
        self.gpu_memory_peak = np.zeros((self.nworkers + 1))

        # run workers in parallel
        for comm in self.comms:
            comm.send('reset_gpu_mem')
            comm.send(['run_batch', epoch])

        # run its own trainer
        batch, stat = self.trainer.run_batch(epoch)
        self.trainer.optimizer.zero_grad(set_to_none=False)
        s = self.trainer.compute_grad(batch)
        merge_stat(s, stat)

        # check if workers are finished
        for comm in self.comms:
            s = comm.recv()
            merge_stat(s, stat)

        # add gradients of workers
        self.obtain_grad_pointers()
        for i in range(len(self.grads)):
            for g in self.worker_grads:
                self.grads[i] += g[i]
            self.grads[i] /= stat['num_steps']

        self.trainer.optimizer.step()

        for comm in self.comms:
            comm.send('get_gpu_mem')
            id, mem = comm.recv()
            self.update_mem_peak(id + 1, *mem)

        return stat, np.array(self.cpu_memory_peak), np.array(self.gpu_memory_peak)

    def state_dict(self):
        return self.trainer.state_dict()

    def load_state_dict(self, state):
        self.trainer.load_state_dict(state)

    def rng_states(self):
        for comm in self.comms:
            comm.send('get_rng')
        return [capture_rng()] + [comm.recv() for comm in self.comms]

    def restore_rng_states(self, states):
        if len(states) != self.nworkers + 1:
            raise ValueError('Recovery collector count differs')
        for comm, state in zip(self.comms, states[1:]):
            comm.send(['set_rng', state])
        for comm in self.comms:
            if comm.recv() is not True:
                raise RuntimeError('Collector did not acknowledge RNG recovery')
        restore_rng(states[0])

    def collector_resources(self):
        for comm in self.comms:
            comm.send('get_resources')
        return [process_resources()] + [comm.recv() for comm in self.comms]

    def reset_mem_peak(self):
        self.cpu_memory_peak = np.zeros((self.nworkers + 1))
        self.gpu_memory_peak = np.zeros((self.nworkers + 1))

        self.trainer.reset_memory_peak()

    def update_mem_peak(self, id, cpu_mem, gpu_mem):
        if cpu_mem > self.cpu_memory_peak[id]:
            self.cpu_memory_peak[id] = cpu_mem

        if gpu_mem > self.gpu_memory_peak[id]:
            self.gpu_memory_peak[id] = gpu_mem
