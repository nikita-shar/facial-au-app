import os
# import wandb
# import logging
# import numpy as np
# import scipy.sparse as sp
# from datetime import datetime
from typing import Optional, List
# from sklearn.metrics import f1_score
# from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import Linear
# from torch_geometric.loader import DataLoader

# from feafaplus_dataset import FEAFAPlusAU, DataSplit, FEAFAPlusAUGraph, FEAFAGraph

torch.manual_seed(42)


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    #device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class DGCNN(nn.Module):
    def __init__(self, mlp_list: List[int], k: int, aggr: str = 'max'):
        super(DGCNN, self).__init__()
        self.k = k
        self.aggr = aggr 

        # First edge conv layers using graph features
        self.conv_layers = nn.ModuleList()
        self.batchnorms = nn.ModuleList()

        for i in range(len(mlp_list) - 1):
            in_channels = mlp_list[i] * 2 # for edge feature [x_i || x_j - x_i]
            out_channels = mlp_list[i + 1]

            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(negative_slope=0.2)
                )
            )

        #embedding the last value in mlp_list
        emb_dims = mlp_list[-1]
        self.conv5 = nn.Sequential(
            nn.Conv1d(sum(mlp_list[1:]), emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 3) 

    def forward(self, x, batch=None):
        batch_size = x.size(0)
        features = []
        #print(f"DGCNN.forward - input x          = {tuple(x.shape)}")

        for idx, conv in enumerate(self.conv_layers):
            x_graph = get_graph_feature(x, k=self.k)
            #print(f"  - after get_graph_feature [{idx}] = {tuple(x_graph.shape)}")
            x = conv(x_graph)
            #print(f"  - after conv{idx}            = {tuple(x.shape)}")
            x = x.max(dim=-1)[0]
            #print(f"  - after max(dim=-1)          = {tuple(x.shape)}")
            features.append(x)

        x = torch.cat(features, dim=1)
        #print(f"  - after torch.cat(features)  = {tuple(x.shape)}")
        x = self.conv5(x)
        #print(f"  - after conv5 (1d)           = {tuple(x.shape)}")

        batch_size = x.size(0)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        #print(f"  - after adaptive_max_pool1d  = {tuple(x1.shape)}")
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        #print(f"  - after adaptive_avg_pool1d  = {tuple(x2.shape)}")
        x = torch.cat((x1, x2), dim=1)
        #print(f"  - after final torch.cat      = {tuple(x.shape)}")

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        #print(f"  - final output               = {tuple(x.shape)}")
        return x
    


def global_max_pool(x, batch):
    B = batch.max().item() + 1
    out = torch.full((B, x.size(1)), float('-inf'), device=x.device)
    out = out.scatter_reduce(0, batch[:, None].expand(-1, x.size(1)), x, reduce='amax', include_self=True)
    return out


class EdgeCNNMT(nn.Module):
    def __init__(self, k: int = 20, aggr: str = 'max', num_aus: int = 24, num_classes: int = 3):
        super().__init__()
        self.k = k
        self.aggr = aggr

        # block1_mlp = mlp_list[:-2]
        # self.block1 = DGCNN( mlp_list=block1_mlp, k=k, aggr=aggr )
        self.block1 = DGCNN( mlp_list=[3, 64, 64, 64], k=k, aggr=aggr )
        # dim1 = sum(block1_mlp[1:]) 
        # block2_mlp = [ dim1, mlp_list[-1] ] 
        # self.block2 = DGCNN(mlp_list=block2_mlp, k=k, aggr=aggr)
        self.block2 = DGCNN(mlp_list=[64, 128], k=k, aggr=aggr)
        # dim2 = mlp_list[-1] 
        # self.lin1 = nn.Linear(dim1 + dim2, 512)
        self.lin1 = nn.Linear(64 + 128, 512)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            ) for _ in range(num_aus)
        ])


    def forward(self, data, batch = None):
        x = data.pos.reshape(-1, 3, 468)
        B,C,N = x.shape
        #print(f"MT.forward - input x={x.shape}, batch={batch.shape}")

        out = x
        feats1 = []
        for idx, conv in enumerate(self.block1.conv_layers):
            g = get_graph_feature(out, k=self.k)    #[B,2*C, N, k]
            #print(f"  block1 get_graph_feature[{idx}] = {g.shape}")
            out = conv(g).max(dim=-1)[0]               #[B, out_c, N]
            #print(f"  block1 conv{idx} output          = {out.shape}")
            # feats1.append(out)
        # f1 = torch.cat(feats1, dim=1)                  #[B, dim1, N]
        #print(f"  block1 concatenated f1           = {f1.shape}")

        # out = f1
        # feats2 = []
        f1 = out
        for idx, conv in enumerate(self.block2.conv_layers):
            g = get_graph_feature(out, k=self.k)    #[B,2 * dim1, N, k]
            #print(f"  block2 get_graph_feature[{idx}] = {g.shape}")
            out = conv(g).max(dim=-1)[0]               #[B, out_c, N]
            #print(f"  block2 conv{idx} output          = {out.shape}")
            # feats2.append(out)
        # f2 = torch.cat(feats2, dim=1)                  #[B, dim2, N]
        #print(f"  block2 concatenated f2           = {f2.shape}")
        
        cat = torch.cat([f1, out], dim=1)           
        # cat = torch.cat([f1, f2], dim=1)           
        #print(f"  combined cat                     = {cat.shape}")
        pooled = cat.max(dim=2)[0]                    
        #print(f"  pooled over N                    = {pooled.shape}")
        proj = self.lin1(pooled)                    
        #print(f"  after lin1(proj)                 = {proj.shape}")

        out_preds = []
        for i, head in enumerate(self.heads):
            h = head(proj).unsqueeze(2)                
            #print(f"  head[{i}] output                 = {h.shape}")
            out_preds.append(h)
        out = torch.cat(out_preds, dim=2)              
        #print(f"  final out                        = {out.shape}")

        return out
    
    @property
    def name(self):
        return "EdgeCNNMT"

def au_mse_loss(gt: torch.Tensor, pred: torch.Tensor):
    assert len(gt.shape) == 2
    assert gt.shape == pred.shape

    # sum over the aus mean over the batch
    sq_diff = (gt - pred)**2
    idx = gt > 0.0
    zero_idx = idx == False
    sq_diff[idx] *= 10.
    pred_nz_idx = pred > 0.0
    error_nz = zero_idx * pred_nz_idx
    sq_diff[error_nz] *= 2.
    sq_diff_sum = sq_diff.sum(dim=-1)
    return sq_diff_sum.mean()


def au_mse_metric(gt: torch.Tensor, pred: torch.Tensor):
    assert len(gt.shape) == 2
    assert gt.shape == pred.shape

    # sum over the aus mean over the batch
    sq_diff = (gt - pred)**2
    sq_diff_sum = sq_diff.sum(dim=-1)
    return sq_diff_sum.mean()


def focal_loss(gt: torch.Tensor, pred: torch.Tensor,
               alpha: torch.Tensor, gamma: float = 2.):

    ce_loss = F.cross_entropy(pred, gt, reduction='none', weight=alpha)
    pt = torch.exp(-ce_loss)
    floss = ((1 - pt)**gamma * ce_loss)

    return floss.mean()


def calculate_f1(gt: torch.Tensor, pred: torch.Tensor):
    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    num_aus = pred.shape[-1]
    f1_scores = np.array(
        [f1_score(gt_np[:, i], pred_np[:, i], average='weighted') for i in range(num_aus)])
    return f1_scores.mean()


def train(should_val: bool = False):

    @dataclass
    class Hyperparams:
        learning_rate: float = 5e-4
        dropout: float = 0.1
        weight_decay: float = 1e-4

    hyperparams = Hyperparams()

    current_date_time = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")

    filename = os.path.basename(__file__)
    print(filename)
    print(filename.split('.py'))
    logdir = 'logs'
    os.makedirs(logdir, exist_ok=True)
    logfile_name = os.path.join(logdir,
                                filename.split('.py')[0] + '_{}.log'.format(current_date_time))

    logging.basicConfig(filename=logfile_name, filemode='w')
    logfile_handle = None
    if not os.path.exists(logfile_name):
        print(f'Log file {logfile_name} not created')
        log = print
        # logfile_handle = open(logfile_name, 'w')
        # log = lambda s : logfile_handle.write(s+'\n')
    else:
        log = logging.getLogger()
        log.setLevel(logging.INFO)

    is_distillation = False
    is_classification = False
    annotations_file = 'opengraphAU_labels.csv'
    reference_face = np.loadtxt('./mean_face.landmark', delimiter=',')
    reference_face = torch.from_numpy(reference_face)
    # train_data = FEAFAPlusAUGraph(annotations_file,
    #                               oau_distillation=is_distillation,
    #                               classification=is_classification)
    log.info('Loading data\n')
    train_data = FEAFAGraph(reference_face, DataSplit.TRAIN.value)

    train_loader = DataLoader(train_data, batch_size=64,
                              shuffle=True, drop_last=False)

    # test_data = FEAFAPlusAUGraph(annotations_file, split=DataSplit.TEST,
    #                              oau_distillation=is_distillation,
    #                              classification=is_classification)
    test_data = FEAFAGraph(reference_face, DataSplit.TEST.value)
    test_loader = DataLoader(test_data, batch_size=64,
                             shuffle=True, drop_last=False)

    # val_data = FEAFAPlusAUGraph(annotations_file, split=DataSplit.VAL,
    #                             oau_distillation=is_distillation,
    #                             classification=is_classification)
    val_data = FEAFAGraph(reference_face, DataSplit.VAL.value)
    val_loader = DataLoader(val_data, batch_size=64,
                            shuffle=True, drop_last=False)

    log.info('Finished loading data')
    log.info(f'Training data has {len(train_data)} samples')

    # gcn = AUGCN(in_channels_list=channels_list, num_classes=24)

    # gcn = EdgeCNNMT(24)
    gcn = EdgeCNNMT()
    # gcn = EdgeCNNMT(len(train_data.au_idxs))
    # gcn = EdgeCNN(24)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    gcn = gcn.to(device)

    optimizer = optim.Adam(gcn.parameters(), lr=hyperparams.learning_rate)
    scheduler = optim.lr_scheduler.\
        ReduceLROnPlateau(optimizer, 'max', min_lr=1e-6)
    # optimizer = optim.SGD(gcn.parameters(), lr = 1e-4, weight_decay=0.0)
    # loss = nn.MSELoss(reduction='sum')
    # loss = au_mse_loss
    class_weights = torch.tensor([1.1128, 12.4940, 46.9536])

    # per_au_class_weights = torch.tensor([[0.5695, 0.5808, 0.9727, 0.9729,
    #                                       0.8687, 0.8695, 0.9472, 0.9454,
    #                                       0.9373, 0.9867, 0.9855, 0.7862,
    #                                       0.7891, 0.9661, 0.9664, 0.9480,
    #                                       0.9513, 0.9894, 0.8505, 0.8054,
    #                                       0.9656, 0.9601, 0.9821, 0.9719],
    #                                      [0.3519, 0.3410, 0.0156, 0.0154,
    #                                       0.1000, 0.0990, 0.0355, 0.0370,
    #                                       0.0390, 0.0082, 0.0089, 0.1885,
    #                                       0.1875, 0.0292, 0.0288, 0.0344,
    #                                       0.0295, 0.0069, 0.1199, 0.1679,
    #                                       0.0259, 0.0253, 0.0060, 0.0199],
    #                                      [0.0786, 0.0783, 0.0118, 0.0118,
    #                                       0.0313, 0.0314, 0.0173, 0.0177,
    #                                       0.0237, 0.0052, 0.0055, 0.0253,
    #                                       0.0234, 0.0048, 0.0048, 0.0176,
    #                                       0.0192, 0.0038, 0.0296, 0.0267,
    #                                       0.0085, 0.0146, 0.0120, 0.0082]])

    per_au_class_weights = torch.tensor(
        [[1.,           1.29397428,   5.4276758],
         [1.,           1.36701913,   5.55365509],
         [1.,          51.05836139,  71.8972738],
         [1.,          51.54488813,  71.91031213],
         [1.,           7.56819361,  24.8833566],
         [1.,           7.66124571,  24.87472902],
         [1.,          21.76012846,  46.34832939],
         [1.,          20.9143367,  45.34046944],
         [1.,          21.09744273,  30.32137258],
         [1.,         108.86024662, 190.1425641],
         [1.,         100.98309706, 179.80873786],
         [1.,           3.59415078,  26.58575149],
         [1.,           3.6176938,  28.58461232],
         [1.,          28.20352739, 188.22916667],
         [1.,          28.92736, 177.07737512],
         [1.,          24.1934117,  47.17457356],
         [1.,          28.57034893,  42.61909331],
         [1.,         138.42516754, 226.99023199],
         [1.,           6.25810056,  25.38080596],
         [1.,           4.10547236,  25.84913641],
         [1.,          33.98197522, 103.24472333],
         [1.,          32.92058824,  50.60412546],
         [1.,         160.55972101,  66.77374909],
         [1.,          43.57723577, 110.78419453]])

    if torch.cuda.is_available():
        class_weights = class_weights.to('cuda')
        per_au_class_weights = per_au_class_weights.to('cuda')

    loss = nn.CrossEntropyLoss(weight=class_weights)
    num_epochs = 100
    n_parameters = sum(p.numel() for p in gcn.parameters() if p.requires_grad)

    log.info(f'Num of parameters {(n_parameters/1e6):.2f} M')
    best_val_loss = 10000.
    best_val_acc = -10000.

    # start a new wandb run to track this script
    wandb_cfg = {
        "model": gcn.name,
        "dataset": "FEAFA+",
        "epochs": num_epochs,
        "architecture": repr(gcn),
        "oau_distillation": is_distillation,
        "classification": is_classification,
        "resume": False,
        "cfg": "EdgeCNN intensity regression + reduce lr on plateau"
    }

    for k, v in asdict(hyperparams).items():
        wandb_cfg[k] = v

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="LM2AU",

    #     # track hyperparameters and run metadata
    #     config=wandb_cfg)

    start_epoch = 0
    if wandb_cfg['resume']:
        log.info('Loading checkpoint and resuming training')
        checkpoint_file = './LM2AU/FaceLandmarkPyTorch/checkpoints/2024-07-10 10:14:22.919969/checkpoint_58.tar'
        loaded_data = torch.load(checkpoint_file, map_location=device)
        gcn.load_state_dict(loaded_data['model_state_dict'])
        scheduler_state_dict = loaded_data['scheduler_state_dict']
        scheduler.load_state_dict(scheduler_state_dict)
        optimizer = optim.Adam(
            gcn.parameters(), lr=scheduler_state_dict['_last_lr'][0])
        start_epoch = scheduler_state_dict['last_epoch']
        log.info("Resuming the training")

    log.info(f'Training with the config {wandb_cfg}')
    for epoch in range(start_epoch, num_epochs):
        gcn.train()
        with torch.autograd.anomaly_mode.set_detect_anomaly(True):
            running_loss = 0.0
            num_train_correct = 0
            num_train_samples = 0
            for it, sample in enumerate(train_loader):
                sample = sample.to(device)
                pred_aus = gcn(sample)
                gt_aus = sample.y.to(pred_aus).long()
                # per_au_loss = [F.cross_entropy(input=pred_aus[:, :, i],
                #                                target=gt_aus[:, i],
                #                                weight=
                #                                 1./per_au_class_weights[:, i],
                #                                reduction='mean')
                #                                for i in range(24)]

                # per_au_loss = [focal_loss(gt=gt_aus[:, i],
                #                           pred=pred_aus[:, :, i],
                #                           alpha=1./per_au_class_weights[:, i]) for i in range(24)]

                per_au_loss = [focal_loss(gt=gt_aus[:, i],
                                          pred=pred_aus[:, :, i],
                                          alpha=per_au_class_weights[i, :]) for i in range(pred_aus.shape[-1])]
                l = sum(per_au_loss)
                # l = loss(input=pred_aus, target=gt_aus)
                # l = au_mse_loss(pred=pred_aus, gt=gt_aus)
                # l = loss(pred_aus, au)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                pred_labels = torch.argmax(
                    torch.log_softmax(pred_aus, dim=1), dim=1)
                # num_train_correct += (pred_labels == gt_aus).sum()
                # num_train_samples += pred_labels.numel()
                # metric = (num_train_correct/num_train_samples) * 100.
                # metric = au_mse_metric(gt=gt_aus, pred=pred_aus)
                # metric = calculate_f1(gt_aus, pred_labels)
                running_loss += l.item()
                # running_loss += metric

                if it % 100 == 0:
                    msg = f'{epoch} : {it} : Loss {l:.3f} : AvgLoss {running_loss/(it+1)}'
                    # msg = f'{epoch} : {it} : Loss {l:.3f} : AvgAcc {metric}'
                    print(msg)
                    log.info(msg)

            # wandb.log({"Train_RL": running_loss})
            # wandb.log({"Train_Acc": metric})

            if should_val:
                val_loss = 0.0
                num_correct = 0
                num_samples = 0
                log.info('===> Starting Validation <===')
                gcn.eval()
                val_f1s = []
                with torch.no_grad():
                    for vit, sample in enumerate(val_loader):
                        sample = sample.cuda()
                        pred_aus = gcn(sample)
                        pred_aus = torch.argmax(
                            F.log_softmax(pred_aus, dim=1), dim=1)
                        # pred_aus = gcn(lm)
                        gt_aus = sample.y.to(pred_aus).long()
                        # num_correct += (pred_aus == gt_aus).sum()
                        # num_samples += gt_aus.numel()
                        # l = loss(pred_aus, au)
                        # l = au_mse_metric(pred=pred_aus, gt=gt_aus)
                        # val_loss += l.item()
                        val_f1 = calculate_f1(gt_aus, pred_aus)
                        val_f1s.append(val_f1)
                        if vit % 100 == 0:
                            # msg = f'{epoch} : {vit} : AvgValLoss {val_loss/(vit+1)}'
                            # msg = f'{epoch} : {vit} : Accuracy {num_correct/num_samples}'
                            msg = f'{epoch} : {vit} : Accuracy {val_f1}'
                            print(msg)
                            log.info(msg)
                    if np.mean(val_f1s) > best_val_acc:
                        best_val_acc = np.mean(val_f1s)
                        log.info(f'{epoch} has val acc {best_val_acc}')
                    # scheduler.step(val_acc)
                    # if val_loss < best_val_loss:
                    #     best_val_loss = val_loss
                    #     log(f'{epoch} has val loss {best_val_loss}')
                    # scheduler.step(-val_loss)
            # wandb.log({"Val_RL": val_loss})
            # wandb.log({"Val_Acc": val_acc})

            if logfile_handle is not None:
                logfile_handle.flush()
            checkpoint_dir = 'checkpoints/{}'.format(current_date_time)
            os.makedirs(checkpoint_dir, exist_ok=True)
            log.info(f'=> saving checkpoint to'
                f'{checkpoint_dir}/checkpoint_{epoch}.tar')
            states = dict()
            states['model_state_dict'] = gcn.state_dict()
            states['optimizer_state_dict'] = optimizer.state_dict()
            states['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(states, os.path.join(checkpoint_dir,
                                            f'checkpoint_{epoch}.tar'))
        if logfile_handle is not None:
            logfile_handle.close()


if __name__ == '__main__':
    train(True)
