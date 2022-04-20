import argparse, os
import transformations as ts
import opt_tc as tc
import numpy as np
import torch
from data_loader import Data_Loader

def transform_data(data, trans):
    trans_inds = np.tile(np.arange(trans.n_transforms), len(data))
    trans_data = trans.transform_batch(np.repeat(np.array(data), trans.n_transforms, axis=0), trans_inds)
    return trans_data, trans_inds

def load_trans_data(args, trans):
    name = "trans_data/data_cls" + str(args.class_ind) + '.pth'
    if os.path.isfile(name):
        x_train_trans, labels, x_test_trans, y_test = torch.load(name)
    else:
        dl = Data_Loader()
        x_train, x_test, y_test = dl.get_dataset(args.dataset, true_label=args.class_ind)
        x_train_trans, labels = transform_data(x_train, trans)
        x_test_trans, _ = transform_data(x_test, trans)
        torch.save([x_train_trans, labels, x_test_trans, y_test],name,pickle_protocol = 4)
    print(len(labels))

    #move axis
    x_test_trans, x_train_trans = x_test_trans.transpose(0, 3, 1, 2), x_train_trans.transpose(0, 3, 1, 2)
    
    y_test = np.array(y_test) == args.class_ind
    return x_train_trans, x_test_trans, y_test, 
#y_test = 

def train_anomaly_detector(args):
    transformer = ts.get_transformer(args.type_trans)
    x_train, x_test, y_test = load_trans_data(args, transformer)
    tc_obj = tc.TransClassifier(transformer.n_transforms, args)
    tc_obj.fit_trans_classifier(x_train, x_test, y_test)
    name = "Net_Class_"+str(args.class_ind)+".pth"
    torch.save(tc_obj.netWRN.state_dict(),name)





if __name__ == '__main__':
    results_list =''
    parser = argparse.ArgumentParser(description='Wide Residual Networks')
    # Model options
    parser.add_argument('--depth', default=10, type=int)
    parser.add_argument('--widen-factor', default=8, type=int)

    # Training options
    parser.add_argument('--batch_size', default=1000, type=int)
    #parser.add_argument('--batch_size', default=288, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--epochs', default=16, type=int)

    # Trans options
    parser.add_argument('--type_trans', default='simple', type=str)

    # CT options
    parser.add_argument('--lmbda', default=0.1, type=float)
    parser.add_argument('--m', default=1, type=float)
    parser.add_argument('--reg', default=True, type=bool)
    parser.add_argument('--eps', default=0, type=float)

    # Exp options
    parser.add_argument('--class_ind', default=1, type=int)
    parser.add_argument('--dataset', default='cifar10', type=str)
    args = parser.parse_args()
    
    for i in range(10):
        args.class_ind = i
        print("Dataset: CIFAR10")
        print("True Class:", args.class_ind)
        train_anomaly_detector(args)
        results_list = "Dataset: CIFAR10"+"True Class:"+ str(args.class_ind)+"\n"
        file_name ="results_CIFAR10_MCR2_10.txt"
        with open(file_name, 'a') as f:
            f.write(results_list )
    
        
  


  
