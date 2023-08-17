import os
import numpy as np
import torch
import torchvision
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from simclr.modules.sync_batchnorm import convert_model

from model import load_optimizer, save_model
from utils import yaml_config_hook
from dataset import dataset_DFD
# from dataset_single_MAE import dataset_DFD

# ori
# from simclr.modules import SingleViT2 as VViT

# rearrange
from simclr.modules.STViT_rearrange import SingleViT2 as VViT
import tqdm
from pdb import set_trace as sts

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.multiprocessing.set_sharing_strategy('file_system')

l1loss = torch.nn.L1Loss()
def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, (x,y) in enumerate(train_loader):
        optimizer.zero_grad()
        y=y.to('cuda')
        # positive pair, with encoding
        pred, rec_seq = model(x)
        loss1 = criterion(pred, y)
        if args.masked_reconstruction:
            loss2 = l1loss(rec_seq, x['ori_img'].cuda())
            loss2 /= args.FRR
            loss = loss1 + args.loss2_weight*loss2
        else:
            loss = loss1
        loss.backward()


        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.local_rank == 0 and step % args.view_interval == 0:
            acc = np.mean((torch.argmax(pred,1)==y).cpu().numpy())
            if args.masked_reconstruction:
                print(f"Step [{step}/{len(train_loader)}][Epoch: {args.current_epoch}]\t Cls-Loss: {loss1.item()}\t Rec-Loss: {loss2.item()}\t Accuracy: {acc}")
                writer.add_scalar('Training/Rec-loss', loss2.item(),  args.global_step)
            else:
                print(f"Step [{step}/{len(train_loader)}][Epoch: {args.current_epoch}]\t Cls-Loss: {loss1.item()}\t Accuracy: {acc}")
            writer.add_scalar('Training/Accuracy', acc,  args.global_step)
            writer.add_scalar('Training/Cls-loss', loss1.item(),  args.global_step)

        if args.local_rank == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def main(rank, args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("#"*80)
    print(f"Start training model on {args.train_file}")
    print(f"Testing on {args.test_file}")

    print("#"*80)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if args.lm != "":
        args.lbp_method = args.lm
        
    train_dataset = dataset_DFD(args,mode = 'train') 
    test_dataset = dataset_DFD(args,mode = 'test') 


    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size if args.test_aug!=1 else 1,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers if args.test_aug!=1 else 1,
    )


    model = VViT('cuda:0', args)
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, args.pos_weight]).to(args.device))

    # DDP / DP
#     if args.dataparallel:
#         model = convert_model(model)
#         model = DataParallel(model)
#     else:
#         if args.nodes > 1:
#             model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#             model = DDP(model, device_ids=[rank], output_device=rank)

    args.current_epoch = 0
    if args.reload:

        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.pth".format(args.epoch_num)
        )
        checkpoint = torch.load(model_fp, map_location=args.device.type)
        new_ckpt = {}
        for k,v in checkpoint['model_state_dict'].items():
            new_ckpt[k.replace('module.', '')] = v
        model.load_state_dict(new_ckpt)
        
#         model.load_state_dict()
        args.current_epoch = args.epoch_num

    model = model.to(args.device)
    writer = None
    if rank == 0:
        writer = SummaryWriter("runs/%s" % args.model_path)

    args.global_step = 0

    for epoch in range(args.current_epoch, args.epochs+1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        lr = optimizer.param_groups[0]["lr"]
        
        # Validation
        if epoch% 10==2:
            model.eval()

            with torch.no_grad():
                accset = []
                for x,y in tqdm.tqdm(test_loader,total=len(test_loader)):
                    if args.test_aug==1:

                        y = y.to('cuda')
                        pred=model(x, forward_mode='test')
                        pred=torch.stack(pred)
                        pred=torch.mean(pred, 0)
                        acc1 = np.mean((torch.argmax(pred,1)==y).cpu().numpy())
                        accset.append(acc1)
                    else:  

                        y=y.to('cuda')
                        pred = model(x, forward_mode='test')
                        acc1 = np.mean((torch.argmax(pred,1)==y).cpu().numpy())
                        accset.append(acc1)

                if args.local_rank==0:
                    print('-'*80)
                    print('Validation accuracy:', np.mean(np.array(accset)))
                    writer.add_scalar("Val/accuracy", np.mean(np.array(accset)), epoch)
                    print('-'*80)
            model.train()
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)
        if rank == 0 and scheduler:
            scheduler.step()

        if rank == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{args.current_epoch}/{args.epochs}]\t Epoch-Loss: {loss_epoch / len(train_loader)}\t lr: {lr}"
            )
            args.current_epoch =epoch
            
        if rank == 0 and epoch % 10 == 0:
            save_model(args, model, optimizer, dict_only=True)

    ## end training
    save_model(args, model, optimizer, filename='last_c23_STFE_FE_MA.pth', dict_only=True)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="DFD")
    config = yaml_config_hook("./config/config_single.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--lm", type=str, default="")
    parser.add_argument("--exp", type=int, default=None)
    args = parser.parse_args()
    print(args)
    


    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"



    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes
    # condition failed
    if args.nodes > 1:
        dist.init_process_group("nccl", rank=args.local_rank, world_size=args.world_size)
        torch.cuda.set_device(args.local_rank)
        local_rank = torch.distributed.get_rank()
        print('args.local_rank: ', local_rank)
        args.device = torch.device("cuda:%d"%local_rank if torch.cuda.is_available() else "cpu")
        main(local_rank, args)
    else:
        print('args.local_rank: ', args.local_rank)
        args.local_rank = 0
        main(0, args)
    
