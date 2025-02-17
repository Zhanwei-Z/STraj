# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import numpy as np
import random
import sys
import time
import shutil
from importlib import import_module
from numbers import Number

from tqdm import tqdm
import torch
from torch.utils.data import Sampler, DataLoader
import horovod.torch as hvd


from torch.utils.data.distributed import DistributedSampler

from utils import Logger, load_pretrain

from mpi4py import MPI


comm = MPI.COMM_WORLD
hvd.init()
torch.cuda.set_device(hvd.local_rank())

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
parser.add_argument(
    "-m", "--model", default="lanegcn", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true")
parser.add_argument(
    "--resume", default="", type=str, metavar="RESUME", help="checkpoint path"
)
parser.add_argument(
    "--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
)


def main():
    seed = hvd.rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model()

    if config["horovod"]:
        opt.opt = hvd.DistributedOptimizer(
            opt.opt, named_parameters=net.named_parameters()
        )

    if args.resume or args.weight:
        ckpt_path = args.resume or args.weight
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(config["save_dir"], ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        load_pretrain(net, ckpt["state_dict"])
        if args.resume:
            config["epoch"] = ckpt["epoch"]
            opt.load_state_dict(ckpt["opt_state"])

    if args.eval:
        # Data loader for evaluation
        dataset = Dataset(config["val_split"], config, train=False)
        val_sampler = DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank()
        )
        val_loader = DataLoader(
            dataset,
            batch_size=config["val_batch_size"],
            num_workers=config["val_workers"],
            sampler=val_sampler,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        hvd.broadcast_parameters(net.state_dict(), root_rank=0)
        val(config, val_loader, net, loss, post_process, 999)
        return

    # Create log and copy all code
    save_dir = config["save_dir"]
    log = os.path.join(save_dir, "log")
    if hvd.rank() == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sys.stdout = Logger(log)

        src_dirs = [root_path]
        dst_dirs = [os.path.join(save_dir, "files")]
        for src_dir, dst_dir in zip(src_dirs, dst_dirs):
            files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            for f in files:
                shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    # Data loader for src training
    dataset = Dataset(config["train_split"], config, train='t_s')
    train_sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    print('Data loader for src training is finished')

    # Data loader for tar training
    dataset = Dataset(config["train_split"], config, train='t_t')
    train_sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    train_loader_tar = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    print('Data loader for tar training is finished')

    # Data loader for src evaluation
    dataset = Dataset(config["val_split"], config, train='f_s')
    val_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        sampler=val_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print('Data loader for src evaluation is finished')

    # Data loader for tar evaluation
    dataset = Dataset(config["val_split"], config, train='f_t')
    val_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader_tar = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        sampler=val_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print('Data loader for tar evaluation is finished')


    hvd.broadcast_parameters(net.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(opt.opt, root_rank=0)

    epoch = config["epoch"]
    remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))
    for i in range(remaining_epochs):
        train(epoch + i, config, train_loader, train_loader_tar, net, loss, post_process, opt, val_loader, val_loader_tar)


def worker_init_fn(pid):
    np_seed = hvd.rank() * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


def actor_aug(data, config):
    data['feats_w'] = [None] * len(data["feats"])  # =batch size
    data['feats_s'] = [None] * len(data["feats"])  # =batch size
    data['ctrs_w'] = [None] * len(data["ctrs"])  # =batch size
    data['ctrs_s'] = [None] * len(data["ctrs"])  # =batch size
    for i in range(len(data["feats"])):
        if data['feats'][i].shape[0] == 1:
            data['feats_w'][i] = data['feats'][i]
            data['feats_s'][i] = data['feats'][i]
            data['ctrs_w'][i] = data['ctrs'][i]
            data['ctrs_s'][i] = data['ctrs'][i]
        else:
            aug_number = data['feats'][i][1:].shape[0] - int(config['aug_pct'] * (data['feats'][i][1:].shape[0]))
            w_aug_idx = torch.argsort(torch.linalg.norm(data['ctrs'][i][1:], axis=1))[:aug_number]
            s_aug_idx = torch.argsort(torch.linalg.norm(data['ctrs'][i][1:], axis=1))[-aug_number:]
            data['feats_w'][i] = torch.cat([data['feats'][i][:1], data['feats'][i][1:][w_aug_idx]], axis=0)
            data['feats_s'][i] = torch.cat([data['feats'][i][:1], data['feats'][i][1:][s_aug_idx]], axis=0)
            data['ctrs_w'][i] = torch.cat([data['ctrs'][i][:1], data['ctrs'][i][1:][w_aug_idx]], axis=0)
            data['ctrs_s'][i] = torch.cat([data['ctrs'][i][:1], data['ctrs'][i][1:][s_aug_idx]], axis=0)


def train(epoch, config, train_loader, train_loader_tar, net, loss, post_process, opt, val_loader, val_loader_tar):
    train_loader.sampler.set_epoch(int(epoch))
    net.train()

    num_batches = len(train_loader)
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config["save_freq"] * num_batches))
    display_iters = int(
        config["display_iters"] / (hvd.size() * config["batch_size"])
    )
    val_iters = int(config["val_iters"] / (hvd.size() * config["batch_size"]))
    start_time = time.time()
    metrics = dict()
    train_loader_tar1 = iter(train_loader_tar)
    for j, data in tqdm(enumerate(train_loader),disable=hvd.rank()):
        epoch += epoch_per_batch
        data = dict(data)
        data['type'] = 'src'
        try:
            data_tar = dict(train_loader_tar1.next())
            data_tar['type'] = 'tar'
        except StopIteration:
            train_loader_tar1 = iter(train_loader_tar)
            data_tar = dict(train_loader_tar1.next())
            data_tar['type'] = 'tar'


        actor_aug(data, config)
        actor_aug(data_tar, config)
        output_tar = net(data_tar)
        loss_tar = loss(output_tar, data_tar)
        output = net(data)
        loss_out, loss_out_sw, loss_out_ws = loss(output, data)
        post_out = post_process(output['out'], data)
        post_process.append(metrics, loss_out, post_out)

        opt.zero_grad()
        (loss_tar*0.1 + loss_out["loss"]+loss_out_sw["loss"]*0.1+loss_out_ws["loss"]*0.1).backward()
        lr = opt.step(epoch)

        num_iters = int(np.round(epoch * num_batches))
        if hvd.rank() == 0 and (
            num_iters % save_iters == 0 or epoch >= config["num_epochs"]
        ):
            save_ckpt(net, opt, config["save_dir"], epoch)

        if num_iters % display_iters == 0:
            dt = time.time() - start_time
            metrics = sync(metrics)
            if hvd.rank() == 0:
                post_process.display(metrics, dt, epoch, lr=lr)
            start_time = time.time()
            metrics = dict()

        if num_iters % val_iters == 0:
            val(config, val_loader, net, loss, post_process, epoch, flag='src')
            val(config, val_loader_tar, net, loss, post_process, epoch, flag='tar')

        if epoch >= config["num_epochs"]:
            val(config, val_loader, net, loss, post_process, epoch, flag='src')
            val(config, val_loader_tar, net, loss, post_process, epoch, flag='tar')
            return


def val(config, data_loader, net, loss, post_process, epoch, flag=None):
    net.eval()

    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(data_loader):
        data = dict(data)
        data['type'] = 'val'
        with torch.no_grad():
            output = net(data)
            loss_out = loss(output, data)
            post_out = post_process(output, data)
            post_process.append(metrics, loss_out, post_out)

    dt = time.time() - start_time
    metrics = sync(metrics)
    if hvd.rank() == 0:
        post_process.display(metrics, dt, epoch, flag=flag)
    net.train()


def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict()},
        os.path.join(save_dir, save_name),
    )


def sync(data):
    data_list = comm.allgather(data)
    data = dict()
    for key in data_list[0]:
        if isinstance(data_list[0][key], list):
            data[key] = []
        else:
            data[key] = 0
        for i in range(len(data_list)):
            data[key] += data_list[i][key]
    return data


if __name__ == "__main__":
    main()
