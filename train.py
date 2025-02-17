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
import pickle
import math
from importlib import import_module
from numbers import Number

from tqdm import tqdm
import torch
from torch.utils.data import Sampler, DataLoader
import horovod.torch as hvd


from torch.utils.data.distributed import DistributedSampler

from utils import Logger, load_pretrain

from mpi4py import MPI
from sklearn.metrics.pairwise import cosine_similarity



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
    config, Dataset, collate_fn, net, loss, post_process, opt,  contra_loss = model.get_model()

    if config["horovod"]:
        opt.opt = hvd.DistributedOptimizer(
            opt.opt, named_parameters=net.named_parameters()
        )

    if args.resume or args.weight:
        ckpt_path = args.resume or args.weight
        if args.weight:
            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        elif not os.path.isabs(ckpt_path):
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

    # pseudo label generation

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
        drop_last=False,
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
    mem_ps = {}
    for i in range(remaining_epochs):
        dict_ps = ps_generate(train_loader_tar, net, save_dir, epoch + i)
        thre_ps = []
        if i == 0:
            mem_ps = dict_ps
            for j in mem_ps:
                mem_ps[j]['flag'] = True
                mem_ps[j]['cos_e'] = 1
                mem_ps[j]['l2_e'] = 0
                thre_ps.append(mem_ps[j]['cls'])
        else:
            for j in mem_ps.keys():
                mem_ps[j]['cos'] = cosine_similarity(dict_ps[j]['reg'].reshape(dict_ps[j]['reg'].shape[0], -1),
                                                     mem_ps[j]['reg'].reshape(mem_ps[j]['reg'].shape[0], -1))
                if mem_ps[j]['cos'].max() < math.sqrt(3)/2:
                    mem_ps[j]['flag'] = False
                    mem_ps[j]['cos_e'] = -1
                    mem_ps[j]['l2_e'] = 10000
                    mem_ps[j]['reg'] = np.concatenate([mem_ps[j]['reg'], dict_ps[j]['reg']], axis=0)
                    if type(mem_ps[j]['cls']) != list:
                        mem_ps[j]['cls'] = [mem_ps[j]['cls']]
                    mem_ps[j]['cls'].append(dict_ps[j]['cls'])
                else:
                    mem_ps[j]['flag'] = True
                    idx = mem_ps[j]['cos'].argmax()
                    mem_ps[j]['cos_e'] = mem_ps[j]['cos'].max()
                    mem_ps[j]['l2_e'] = np.linalg.norm(mem_ps[j]['reg'][idx]-dict_ps[j]['reg'][0, ...])
                    if type(mem_ps[j]['cls']) != list:
                        mem_ps[j]['cls'] = [mem_ps[j]['cls']]
                    if mem_ps[j]['cls'][idx] < dict_ps[j]['cls']:
                        mem_ps[j]['cls'] = dict_ps[j]['cls']
                        mem_ps[j]['reg'] = dict_ps[j]['reg']
                    else:
                        mem_ps[j]['cls'] = mem_ps[j]['cls'][idx]
                        mem_ps[j]['reg'] = mem_ps[j]['reg'][idx][np.newaxis]
                    thre_ps.append(mem_ps[j]['cls'])
        if len(thre_ps) > 0:
            threshold_ps = np.array(thre_ps)
            threshold_ps.sort()
            threshold_ps = threshold_ps[-int(len(threshold_ps)*0.2+0.2*i/remaining_epochs)]
        else:
            threshold_ps = None
        train(epoch + i, config, train_loader, train_loader_tar, net, loss,
              post_process, opt, val_loader, val_loader_tar, mem_ps, threshold_ps, contra_loss)

def ps_generate(train_loader_tar, net, save_dir, epoch):
    net.eval()
    dict_ps = {}
    train_loader_tar1 = iter(train_loader_tar)
    for i in range(len(train_loader_tar)):
        data_tar = dict(train_loader_tar1.next())
        data_tar['type'] = 'tar_eval'
        output = net(data_tar)
        for j in range(len(data_tar['idx'])):
            dict_ps[data_tar['idx'][j]] = {}
            dict_ps[data_tar['idx'][j]]['reg'] = output['reg_c'][j][:, output['cls_c'][j].argmax()]
            dict_ps[data_tar['idx'][j]]['cls'] = output['cls_c'][j].max()
    ps_path = os.path.join(save_dir, "pseudo_label")
    dict_ps = comm.allgather(dict_ps)
    for i in range(1, len(dict_ps)):
        for j in dict_ps[i]:
            if j not in dict_ps[0]:
                dict_ps[0][j] = dict_ps[i][j]
    if not os.path.exists(ps_path):
        os.makedirs(ps_path)
    ps_path = os.path.join(ps_path, "ps_epoch_{}.p".format(epoch))
    f = open(ps_path, 'wb')
    pickle.dump(dict_ps[0], f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    print('Pseudo label generation of epoch %d is finished!' % epoch)
    return dict_ps[0]


def worker_init_fn(pid):
    np_seed = hvd.rank() * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


def train(epoch, config, train_loader, train_loader_tar, net, loss,
          post_process, opt, val_loader, val_loader_tar, mem_ps, threshold_ps, contra_loss):
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
    for i, data in tqdm(enumerate(train_loader), disable=hvd.rank()):
        epoch += epoch_per_batch
        data = dict(data)
        data['type'] = 'src'
        try:
            data_tar = dict(train_loader_tar1.next())
            data_tar['type'] = 'tar_train'
        except StopIteration:
            train_loader_tar1 = iter(train_loader_tar)
            data_tar = dict(train_loader_tar1.next())
            data_tar['type'] = 'tar_train'
        output = net(data)
        gt_preds_src = output['gt_preds']
        agents_a_src = output['agents_a']
        loss_out = loss(output, data)
        post_out = post_process(output, data)
        post_process.append(metrics, loss_out, post_out)
        output, agents_a_tar = net(data_tar)
        loss_out_tar = loss(output, data_tar, mem_ps, threshold_ps)

        contra_loss_src = contra_loss(gt_preds_src, agents_a_src)
        contra_loss_tar = contra_loss(output, agents_a_tar)
        conter_loss_st = contra_loss(gt_preds_src, agents_a_src, output, agents_a_tar)
        conter_loss_ts = contra_loss(output, agents_a_tar, gt_preds_src, agents_a_src)
        con_loss = (contra_loss_src + contra_loss_tar + conter_loss_st + conter_loss_ts) * 0.01
        opt.zero_grad()
        (loss_out["loss"]+loss_out_tar + con_loss).backward()
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
