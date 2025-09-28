
#时序补全 Functional tensor meets ode
import torch
from torch import  optim
from tqdm import tqdm
import  matplotlib.pyplot as plt
from model import *
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
#################################### Function defination ##########################################

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)




def get_ind_time(data_dict, key1, key2):
    x1 = torch.tensor(data_dict[key1], dtype=torch.float32) .to(device)
    ind1 = x1[:, :2]
    t1 = x1[:, -1]
    x2 = torch.tensor(data_dict[key2], dtype=torch.long).to(device)
    ind2 = x2[:, :2]
    t2 = x2[:, -1]
    return ind1, ind2, t1, t2




def load_data(data_path, flag, fold):
    if flag:
        full_data = np.load(data_path, allow_pickle=True).item()
        full_data = full_data["data"][fold]
        tr_ind_conti = torch.tensor(full_data["tr_ind_conti"], dtype=torch.float32).to(device)
        tr_ind = torch.tensor(full_data["tr_ind"], dtype=torch.long).to(device)


        tr_time_conti = tr_ind_conti[:, 3]
        tr_time_ind = tr_ind[:, 3]
        tr_ind_conti = tr_ind_conti[:, :3]
        tr_ind = tr_ind[:, :3]


        tr_y = torch.tensor(full_data["tr_y"], dtype=torch.float32).to(device)
        time_uni = torch.tensor(full_data["time_uni"], dtype=torch.float32).to(device)
        u_ind_uni = torch.tensor(full_data["u_ind_uni"], dtype=torch.float32).to(device)
        v_ind_uni = torch.tensor(full_data["v_ind_uni"], dtype=torch.float32).to(device)
        w_ind_uni = torch.tensor(full_data["w_ind_uni"], dtype=torch.float32).to(device)
        return tr_ind_conti, tr_ind, tr_time_conti, tr_time_ind, tr_y, time_uni,  u_ind_uni, v_ind_uni, w_ind_uni
    else:
        full_data = np.load(data_path, allow_pickle=True).item()
        full_data = full_data["data"][fold]
        te_ind_conti = torch.tensor(full_data["te_ind_conti"], dtype=torch.float32).to(device)
        te_ind = torch.tensor(full_data["te_ind"], dtype=torch.long).to(device)

        te_time_conti = te_ind_conti[:, 3]
        te_time_ind = te_ind[:, 3]
        te_ind_conti = te_ind_conti[:, :3]
        te_ind = te_ind[:, :3]

        te_y = torch.tensor(full_data["te_y"], dtype=torch.float32).to(device)
        return te_ind_conti, te_ind, te_time_conti, te_time_ind, te_y


def normalize_data(tr_y, te_y):
    data_mean = tr_y.mean()
    data_std = tr_y.std()
    tr = (tr_y - data_mean) / data_std
    te = (te_y - data_mean) / data_std
    return tr, te, data_mean, data_std







def loss_fn(pred, gt): #RMSE
    # pred, gt: (3000,) tensor
    MSE_loss = torch.nn.MSELoss()
    return torch.sqrt(MSE_loss(pred, gt))


def loss_fn2(pred, gt): #MAE
    MAE_loss = torch.nn.L1Loss()
    return  MAE_loss(pred, gt)


def train(model,  train_loader,  optimizer, loss_fn, epoch):
    model.train()
    loss_list = []
    for i, (train_ind_batch, tr_time_ind_batch, train_y_batch) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        output_batch,  nfe_forward_aug, kl_loss = model(train_ind_batch, tr_time_ind_batch)
        N = train_ind_batch.shape[0]
        c = torch.sigmoid(torch.FloatTensor([(epoch-200)/10])).to(device) #cold start of automatic rank determination mechanism
        loss  = loss_fn(output_batch,  train_y_batch.squeeze()) 
        loss_all = loss + c*kl_loss[0]/N + (1-c)*kl_loss[1]/N
        loss_all.backward()
        optimizer.step()
        loss_list.append(loss.item())
    loss_mean = np.mean(loss_list)
    return loss_mean, nfe_forward_aug



def evaluating(model, test_loader, loss_fn, loss_fn2):
    result = torch.Tensor([]) .to(device)
    labels = []
    print("evaluating....")
    for test_ind_batch, te_time_ind_batch, test_y_batch in test_loader:
        output_batch, nfe_forward, _ = model(test_ind_batch, te_time_ind_batch)
        result = torch.cat((result, output_batch), dim=0)

    rmse = loss_fn(result, te_y.squeeze())*data_std
    mae = loss_fn2(result, te_y.squeeze())*data_std
    return int(rmse.item()*10000)/10000, int(mae.item()*10000)/10000




def train_parallel(model, train_loader, test_loader,  optimizer, loss_fn, loss_fn2, max_iter):
        rmse_min = 10
        rmse, mae = evaluating(model, test_loader, loss_fn, loss_fn2)
        print("Epoach:-1", "Test RMSE:", rmse, " MAE:", mae)
        for epoch in range(max_iter):
            loss, nfe_forward_aug = train(model, train_loader, optimizer, loss_fn, epoch)
            if epoch>10 and epoch % 10 == 0:
                rmse, mae = evaluating(model, test_loader, loss_fn, loss_fn2)
                if rmse < rmse_min:
                    rmse_min = rmse
                print("Epoach", epoch, "Test RMSE:", rmse, " MAE:", mae)
            if epoch % 20 == 0:
                print("Epoch",epoch, ": CATTE loss:", loss, "; nfe_forward:", nfe_forward_aug[0], "; Power of learned factor trajectories:", nfe_forward_aug[1], "; Lambdas:", nfe_forward_aug[2])








if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="SSF")
    parser.add_argument("--batch_size", type=int, default=30000)   
    #parser.add_argument("--data_path", type=str, default=r"./dataset/SSF_10x20x10_fold5.npy") 
    parser.add_argument("--data_path", type=str, default=r"./dataset/traffic_5x20x16_fold5.npy") 
    parser.add_argument("--R", type=int, default=10, help="rank of CP model") 
    parser.add_argument("--J", type=int, default=10, help="dimension of ODE state") 
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--max_iter", type=int, default=5001)
    config = parser.parse_args()



    tr_ind_conti, tr_ind, tr_time_conti, tr_time_ind, tr_y, time_uni,  u_ind_uni, v_ind_uni, w_ind_uni = load_data(config.data_path, flag=1, fold=0)
    te_ind_conti, te_ind, te_time_conti, te_time_ind, te_y = load_data(config.data_path, flag=0, fold=0)
    tr_y, te_y, data_mean, data_std= normalize_data(tr_y, te_y)



    set_random_seed(231)


    model = CATTE_4D(config.J,config.R, time_uni,  u_ind_uni, v_ind_uni, w_ind_uni).to(device)


    optimizer = optim.AdamW(model.parameters(), config.learning_rate)
    print("model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_dataset = TensorDataset(tr_ind, tr_time_ind, tr_y)
    test_dataset = TensorDataset(te_ind, te_time_ind, te_y)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    train_parallel(model, train_loader, test_loader, optimizer, loss_fn, loss_fn2, max_iter=config.max_iter)


