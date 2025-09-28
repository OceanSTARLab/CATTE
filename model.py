#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint
import torch
from torch import nn
dtype = torch.cuda.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")












class ODEfunc_inner(nn.Module):

    def __init__(self, time_dim, uni, mode=0):
        super(ODEfunc_inner, self).__init__()

        self.time_dim = time_dim
        self.uni = uni
        self.mode = mode

        mid_channel = 256

        self.t_net = nn.Sequential(FFLayer(1, 2*self.time_dim,   omega_0=100), nn.Linear(2*self.time_dim, self.time_dim))
        self.ind_net = nn.Sequential(FFLayer(1, 2*self.time_dim, omega_0=100), nn.Linear(2 * self.time_dim, self.time_dim))

        self.fc = nn.Sequential(nn.Linear(3*self.time_dim, mid_channel), nn.ReLU(), nn.Linear(mid_channel, mid_channel), nn.ReLU(), nn.Linear(mid_channel, self.time_dim), nn.Tanh())



    def forward(self, t, x): #(1,1)
        (B,  W) = x.shape
        assert B == self.uni[self.mode].shape[0]
        t_fea = self.t_net(t.unsqueeze(0).unsqueeze(0))
        t_fea = t_fea.expand((B, -1)) # (B, )

        ind_fea = self.ind_net(self.uni[self.mode].unsqueeze(1))


        out = self.fc(torch.concat((x.view(B, -1), t_fea, ind_fea), dim=1)) #fusing index and time information
        return out




class ODEfunc_p_merge_4D(nn.Module): 
        # ODE function for merging 3 different dynamics into one ODE block
    def __init__(self,  dim, uni, uni_num):
        super(ODEfunc_p_merge_4D, self).__init__()
        self.uni = uni
        self.uni_num = uni_num



        self.nfe = 0
        self.nets0 = ODEfunc_inner(time_dim=int(dim), uni=self.uni, mode=0)
        self.nets1 = ODEfunc_inner(time_dim=int(dim), uni=self.uni, mode=1)
        self.nets2 = ODEfunc_inner(time_dim=int(dim), uni=self.uni, mode=2)

    def forward(self, t, x): #(1) (1,2,32)
        self.nfe += 1
        (B, C, W) = x.shape
        assert B == self.uni_num[0]+self.uni_num[1]+self.uni_num[2]
        out0 = self.nets0(t, x[:self.uni_num[0], 0, :])
        out1 = self.nets1(t, x[self.uni_num[0]:self.uni_num[0]+self.uni_num[1], 0, :])
        out2 = self.nets2(t, x[(self.uni_num[0]+self.uni_num[1]):, 0, :])
        out = torch.cat((out0, out1, out2), dim=0)
        return out.view(B, C, W)


class ODEfunc_p_merge_5D(nn.Module): 
        # ODE function for merging 4 different dynamics into one ODE block
    def __init__(self,  dim, uni, uni_num):
        super(ODEfunc_p_merge_5D, self).__init__()
        self.uni = uni
        self.uni_num = uni_num



        self.nfe = 0
        self.nets0 = ODEfunc_inner(time_dim=int(dim), uni=self.uni, mode=0)
        self.nets1 = ODEfunc_inner(time_dim=int(dim), uni=self.uni, mode=1)
        self.nets2 = ODEfunc_inner(time_dim=int(dim), uni=self.uni, mode=2)
        self.nets3 = ODEfunc_inner(time_dim=int(dim), uni=self.uni, mode=3)

    def forward(self, t, x): 
        self.nfe += 1
        (B, C, W) = x.shape
        assert B == self.uni_num[0]+self.uni_num[1]+self.uni_num[2]+self.uni_num[3]


        out0 = self.nets0(t, x[:self.uni_num[0], 0, :])
        out1 = self.nets1(t, x[self.uni_num[0]:self.uni_num[0]+self.uni_num[1], 0, :])
        out2 = self.nets2(t, x[(self.uni_num[0]+self.uni_num[1]):self.uni_num[0]+self.uni_num[1]+self.uni_num[2], 0, :])
        out3 = self.nets3(t, x[(self.uni_num[0]+self.uni_num[1]+self.uni_num[2]):, 0, :])


        out = torch.cat((out0, out1, out2,out3), dim=0)
        return out.view(B, C, W)





class ODEfunc_p_merge_6D(nn.Module): 
        # ODE function for merging 5 different dynamics into one ODE block
    def __init__(self,  dim, uni, uni_num):
        super(ODEfunc_p_merge_6D, self).__init__()
        self.uni = uni
        self.uni_num = uni_num



        self.nfe = 0
        self.nets0 = ODEfunc_inner(time_dim=int(dim), uni=self.uni, mode=0)
        self.nets1 = ODEfunc_inner(time_dim=int(dim), uni=self.uni, mode=1)
        self.nets2 = ODEfunc_inner(time_dim=int(dim), uni=self.uni, mode=2)
        self.nets3 = ODEfunc_inner(time_dim=int(dim), uni=self.uni, mode=3)
        self.nets4 = ODEfunc_inner(time_dim=int(dim), uni=self.uni, mode=4)

    def forward(self, t, x): 
        self.nfe += 1
        (B, C, W) = x.shape
        assert B == self.uni_num[0]+self.uni_num[1]+self.uni_num[2]+self.uni_num[3]+self.uni_num[4]


        out0 = self.nets0(t, x[:self.uni_num[0], 0, :])
        out1 = self.nets1(t, x[self.uni_num[0]:self.uni_num[0]+self.uni_num[1], 0, :])
        out2 = self.nets2(t, x[(self.uni_num[0]+self.uni_num[1]):self.uni_num[0]+self.uni_num[1]+self.uni_num[2], 0, :])
        out3 = self.nets3(t, x[(self.uni_num[0]+self.uni_num[1]+self.uni_num[2]):self.uni_num[0]+self.uni_num[1]+self.uni_num[2]+self.uni_num[3], 0, :])
        out4 = self.nets4(t, x[(self.uni_num[0]+self.uni_num[1]+self.uni_num[2]+self.uni_num[3]):, 0, :])

      
        out = torch.cat((out0, out1, out2,out3, out4), dim=0)
        return out.view(B, C, W)

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc


    def forward(self, x, integration_time): #
        out = odeint(self.odefunc, x, integration_time, rtol=0.0001, atol=0.001, method="dopri5") # method="dopri5"
        #out = odeint(self.odefunc, x, integration_time,  method="euler", options={"step_size": 0.01}) # method="euler"
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class FFLayer(nn.Module):
    # Fourier Feature Layer, 
    # Tancik, Matthew, et al. "Fourier features let networks learn high frequency functions in low dimensional domains." 
    # Advances in neural information processing systems 33 (2020): 7537-7547.
    def __init__(self, in_features, out_features,  omega_0=1):
        super().__init__()
        self.omega_0 = omega_0
        self.in_features = in_features
        self.linear = nn.Linear(in_features, int(0.5*out_features))
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.in_features,1 / self.in_features)

    def forward(self, input):
        x = self.omega_0 * self.linear(input)
        return torch.concat((torch.sin(x), torch.cos(x)), dim=1)





class CATTE_4D(nn.Module):   
    def __init__(self, j, r,  time_uni, u_ind_uni, v_ind_uni, w_ind_uni):
        super(CATTE_4D, self).__init__()
        self.time_uni = time_uni
        self.r_1 = r
        self.r_2 = r
        self.t_num = time_uni.shape[0]
        self.u_ind_uni = u_ind_uni
        self.u_num = u_ind_uni.shape[0]
        self.v_ind_uni = v_ind_uni
        self.v_num = v_ind_uni.shape[0]
        self.w_ind_uni = w_ind_uni
        self.w_num = w_ind_uni.shape[0]
        self.ode_dim = j
        mid_channel_ind = 512
        o = 4
        self.u_ind_net = nn.Sequential(FFLayer(1, mid_channel_ind,  omega_0=o),
                                     nn.Linear(mid_channel_ind, mid_channel_ind, bias=True), nn.ReLU(),
                                     nn.Linear(mid_channel_ind, self.ode_dim, bias=True))



        self.v_ind_net = nn.Sequential(FFLayer(1, mid_channel_ind,  omega_0=o),
                                     nn.Linear(mid_channel_ind, mid_channel_ind, bias=True), nn.ReLU(),
                                     nn.Linear(mid_channel_ind, self.ode_dim, bias=True))

        self.w_ind_net = nn.Sequential(FFLayer(1, mid_channel_ind,  omega_0=1),
                                     nn.Linear(mid_channel_ind, mid_channel_ind, bias=True), nn.ReLU(),
                                     nn.Linear(mid_channel_ind, self.ode_dim, bias=True))



        self.ode_solver = ODEBlock(ODEfunc_p_merge_4D(dim=self.ode_dim, uni=(self.u_ind_uni, self.v_ind_uni, self.w_ind_uni), uni_num=(self.u_num, self.v_num, self.w_num) ))


        self.U_net = nn.Sequential(nn.Linear(self.ode_dim, mid_channel_ind), nn.ReLU(), nn.Linear(mid_channel_ind, mid_channel_ind),
                                   nn.ReLU(), nn.Linear(mid_channel_ind, r))
        self.V_net = nn.Sequential(nn.Linear(self.ode_dim, mid_channel_ind), nn.ReLU(), nn.Linear(mid_channel_ind, mid_channel_ind),
                                   nn.ReLU(), nn.Linear(mid_channel_ind, r))
        self.W_net = nn.Sequential(nn.Linear(self.ode_dim, mid_channel_ind), nn.ReLU(), nn.Linear(mid_channel_ind, mid_channel_ind),
                                   nn.ReLU(), nn.Linear(mid_channel_ind, r))


    
        self.alpha_r = nn.Parameter(torch.abs(torch.ones(r)*1e-1), requires_grad=True)

        self.alpha0 = torch.abs(torch.ones(r)*1e-1).to(device)
        self.beta0 = torch.abs(torch.ones(r)*1e-1).to(device)


        


    def forward(self, train_ind_batch, train_T_batch):
        u_t0 = self.u_ind_net(self.u_ind_uni.unsqueeze(1))
        v_t0 = self.v_ind_net(self.v_ind_uni.unsqueeze(1))
        w_t0 = self.w_ind_net(self.w_ind_uni.unsqueeze(1))

        uvw_t0 = torch.concat((u_t0, v_t0, w_t0), dim=0).unsqueeze(1)

        UVW_fea = self.ode_solver(uvw_t0, self.time_uni)
        nfe_forward = self.ode_solver.nfe
        self.ode_solver.nfe = 0



        U_fea = UVW_fea[:, :self.u_num, :, :]
        V_fea = UVW_fea[:, self.u_num:(self.u_num+self.v_num), :, :]
        W_fea = UVW_fea[:, (self.u_num+self.v_num):, :, :]
        U_fea = U_fea.contiguous()
        V_fea = V_fea.contiguous()
        W_fea = W_fea.contiguous()





        U_fea = self.U_net(U_fea.view(-1, self.ode_dim)).view(self.t_num, self.u_num, -1)  # 100 50 1 
        V_fea = self.V_net(V_fea.view(-1, self.ode_dim)).view(self.t_num, self.v_num, -1)
        W_fea = self.W_net(W_fea.view(-1, self.ode_dim)).view(self.t_num, self.w_num, -1)



        U = U_fea[train_T_batch, train_ind_batch[:, 0], :]
        V = V_fea[train_T_batch, train_ind_batch[:, 1], :]
        W = W_fea[train_T_batch, train_ind_batch[:, 2], :]




        lambda_r = torch.abs(self.alpha_r/self.beta0)


        U_sum2 = torch.sum((U_fea * U_fea).view(-1, self.r_1), dim=0)
        V_sum2 = torch.sum((V_fea * V_fea).view(-1, self.r_1), dim=0)
        W_sum2 = torch.sum((W_fea * W_fea).view(-1, self.r_1), dim=0)

        
        kl_1 = (torch.sum((U_sum2 + V_sum2 + W_sum2) * (lambda_r), dim=0) + torch.sum(torch.log(1 / lambda_r), dim=0)) # sparsity induing term

        kl_2 = torch.sum((torch.abs(self.alpha_r)-self.alpha0)*torch.digamma(self.alpha_r)
                                   -torch.lgamma(torch.abs(self.alpha_r))+torch.lgamma(self.alpha0),dim=0) # prior term
        

        UV = U*V
        out_put = torch.einsum("bi, bi->b", UV, W)
        return out_put,  (nfe_forward, (U_sum2 + V_sum2 + W_sum2).detach().cpu().numpy(), lambda_r.detach().cpu().numpy()), (kl_1,kl_2)





