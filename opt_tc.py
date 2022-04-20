import torch.utils.data
import numpy as np
import torch
import torch.utils.data
from torch.backends import cudnn
from wideresnet import WideResNet
from sklearn.metrics import roc_auc_score
from MCR2 import MaximalCodingRateReduction 

cudnn.benchmark = True

def tc_loss(zs, m):
    means = zs.mean(0).unsqueeze(0)
    res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
    pos = torch.diagonal(res, dim1=1, dim2=2)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).cuda() * 1e6
    neg = (res + offset).min(-1)[0]
    loss = torch.clamp(pos + m - neg, min=0).mean()
    return loss


class TransClassifier():
    def __init__(self, num_trans, args):
        self.n_trans = num_trans
        self.args = args
#         self.netWRN = WideResNet(self.args.depth, num_trans, self.args.widen_factor).cuda()
# Number of trans = 10 We want to have 10 or more dimensions within each trans feature space sub space
#so we do 128 dimension 
        print("number of trans"+str(num_trans))
        self.netWRN = WideResNet(self.args.depth, 32, self.args.widen_factor).cuda()
        self.optimizer = torch.optim.Adam(self.netWRN.parameters())


    def fit_trans_classifier(self, x_train, x_test, y_test):
        print("Training")
        self.netWRN.train()
        bs = self.args.batch_size
        N, sh, sw, nc = x_train.shape
        n_rots = self.n_trans
        m = self.args.m
#         celoss = torch.nn.CrossEntropyLoss()
# Change to MCR2_loss Create objects


        MCR2_loss = MaximalCodingRateReduction(eps=0.2, gamma=0.8)

        
        ndf = 256
        
        #build labels first
#         train_labels_all = torch.from_numpy(np.tile(np.arange(n_rots), x_train.shape[0]//n_rots)).long().cuda()

        for epoch in range(self.args.epochs):
            rp = np.random.permutation(N//n_rots)
            rp = np.concatenate([np.arange(n_rots) + rp[i]*n_rots for i in range(len(rp))])
            assert len(rp) == N
            all_zs = torch.zeros((len(x_train), ndf)).cuda()
            diffs_all = []

            for i in range(0, len(x_train), bs):
                batch_range = min(bs, len(x_train) - i)
                idx = np.arange(batch_range) + i
                xs = torch.from_numpy(x_train[rp[idx]]).float().cuda()
               

                    
                zs_tc, zs_ce = self.netWRN(xs)

#                 all_zs[idx] = zs_tc
                train_labels = torch.from_numpy(np.tile(np.arange(n_rots), batch_range//n_rots)).long().cuda()
#                 zs = torch.reshape(zs_tc, (batch_range//n_rots, n_rots, ndf))
#                 means = zs.mean(0).unsqueeze(0)
#                 diffs = -((zs.unsqueeze(2).detach().cpu().numpy() - means.unsqueeze(1).detach().cpu().numpy()) ** 2).sum(-1)
#                 diffs_all.append(torch.diagonal(torch.tensor(diffs), dim1=1, dim2=2))
#                 tc = tc_loss(zs, m)
#                 ce = celoss(zs_ce, train_labels)
                zs_ce = torch.nn.functional.normalize(zs_ce)
                total_loss=  MCR2_loss(zs_ce, train_labels)
                loss =total_loss[0]
                print(loss)
                
#                 if self.args.reg:
#                     loss = ce + self.args.lmbda * tc + 10 *(zs*zs).mean()
#                 else:
#                     loss = ce + self.args.lmbda * tc
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.netWRN.eval()
            all_zs = torch.reshape(all_zs, (N//n_rots, n_rots, ndf))
            means = all_zs.mean(0, keepdim=True)


            with torch.no_grad():
                batch_size = bs
                train_z_list, train_y_list = [], []
                for i in range(0, len(x_train), bs):
                 
                    batch_range = min(bs, len(x_train) - i)
                    idx = np.arange(batch_range) + i
                    xs = torch.from_numpy(x_train[rp[idx]]).float().cuda()
                    zs_tc, zs_ce = self.netWRN(xs)
                    zs_ce = torch.nn.functional.normalize(zs_ce)
                    train_labels = torch.from_numpy(np.tile(np.arange(n_rots), batch_range//n_rots)).long().cuda()
                    
                    train_z_list.append(zs_ce.clone())
                    train_y_list.append(train_labels.clone())
                    
                train_z = torch.stack(train_z_list,dim=0)
                train_y = torch.stack(train_y_list,dim=0)
                eigv_list = []
                for cls_idx in train_y.unique():
                    sub_z = train_z[train_y==cls_idx]
                    u,s,v = sub_z.svd()
                    s = s/s.max()
                    sig_s = s>0.5
                    eigv_list.append(v.t()[sig_s].clone())
                    
                val_scores = []
#                 print(x_test.shape)
                for i in range(0, len(x_test), batch_size):
                    batch_range = min(batch_size, len(x_test) - i)
                    idx = np.arange(batch_range) + i
                    xs = torch.from_numpy(x_test[idx]).float().cuda()

                    zs, fs = self.netWRN(xs)
                    fs = torch.nn.functional.normalize(fs,dim=1)
                    
                    #calculate score
                    cos_list = []
                    for eig_mtx in eigv_list:
                        fs_proj = fs@eig_mtx.t()@eig_mtx
#                         print(fs_proj.shape)
#                         print(fs.shape)
                        cos_list.append(torch.nn.functional.cosine_similarity(fs_proj,fs,dim=1))
                    cos_all = torch.stack(cos_list,dim=-1).min(-1)[0]
#                     print(cos_all.shape)
                    val_scores.append(torch.stack(cos_list,dim=-1).min(-1)[0])
#                     val_scores[-1].shape
#                 print(len(val_scores))
                val_scores = torch.cat(val_scores,dim=0).cpu().numpy()
                val_scores = -val_scores.reshape((-1,8)).sum(1)
#                 print(val_scores.shape)
                print("Epoch:", epoch, ", AUC: ", roc_auc_score(y_test, val_scores))
                fo = open("results_CIFAR10_MCR2_10.txt","a")
                string_list = ("Epoch:"+ str(epoch) +", AUC: "+ str(roc_auc_score(y_test, val_scores))+"\n")
                fo.write(string_list)


