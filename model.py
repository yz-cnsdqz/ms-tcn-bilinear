from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from torch.nn.parameter import Parameter





class FBM(nn.Module):
    def __init__(self,
                 n_basis,
                 in_dim=64,
                 n_factor=20,
                 use_normalization=True):
        super(FBM, self).__init__()

        self.in_dim = in_dim
        self.out_dim = n_basis
        self.n_factor = n_factor

        self.use_normalization=use_normalization



        Er_np = np.random.standard_normal([self.in_dim, self.n_factor, self.out_dim])
        self.E = Parameter(torch.tensor(Er_np, dtype=torch.float32, 
                                        device=torch.device('cuda:0')),
                            requires_grad=True)


        W_np = np.random.standard_normal([self.in_dim, self.out_dim])
        self.W = Parameter(torch.tensor(W_np, dtype=torch.float32,
                                        device=torch.device('cuda:0')),
                            requires_grad=True)


        
        self.b = Parameter(torch.zeros([1, self.out_dim, 1],
                                        dtype=torch.float32,
                                        device=torch.device('cuda:0')),
                            requires_grad=True)



    def channel_max_normalization(self, x):
        max_vals = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
        return x / (max_vals+1e-5)


    def forward(self, x):

        first_term = torch.einsum('bdt, dc->bct',(x, self.W))
        second_term = torch.einsum('bdt, dkc->bct', (x, self.E))
        out = self.b + first_term + second_term

        if self.use_normalization:
            out = torch.sign(out) * (torch.sqrt(torch.abs(out)+1e-2)-np.sqrt(1e-2))
            out = self.channel_max_normalization(out)

        return out



class RPBinaryPooling(nn.Module):
    def __init__(self, 
                 n_basis=8, 
                 n_rank=4, 
                 use_normalization=True):
        super(RPBinaryPooling, self).__init__()
        self.n_basis = n_basis
        self.n_rank = n_rank
        self.in_dim=64
        self.E_list = nn.ParameterList([])
        self.F_list = nn.ParameterList([])
        self.use_normalization = use_normalization
        np.random.seed(seed=15385)
        for r in range(self.n_rank):
            Er_np = np.sign(np.random.standard_normal([self.in_dim, self.n_basis]))
            self.E_list.append(Parameter(torch.tensor(Er_np,dtype=torch.float32,
                                                      device=torch.device('cuda:0')), 
                                        requires_grad=False)
                            )
                               
            Fr_np = np.sign(np.random.standard_normal([self.in_dim, self.n_basis]))
            self.F_list.append(Parameter(torch.tensor(Fr_np,dtype=torch.float32,
                                                      device=torch.device('cuda:0')), 
                                        requires_grad=False)
                            )
                            

    def channel_max_normalization(self, x):
        max_vals = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
        return x / (max_vals+1e-5)


    def forward(self, x):
        """
        input feature x: [batch, feature_dim, time]
        """
        in_time = x.shape[2]

        z = 0
        for r in range(self.n_rank):
            
#            xer = torch.matmul(x.permute([0,2,1]), self.E_list[r]).permute([0,2,1])
#            xfr = torch.matmul(x.permute([0,2,1]), self.F_list[r]).permute([0,2,1])
#            zr = torch.einsum('bit, bjt->bijt', (xer, xfr)).view(-1, self.n_basis**2, in_time)
#            z += zr
            xer = torch.matmul(x.permute([0,2,1]), self.E_list[r]).unsqueeze(-1)
            xfr = torch.matmul(x.permute([0,2,1]), self.F_list[r]).unsqueeze(-2)
            zr = torch.matmul(xer, xfr).view(-1, in_time, self.n_basis**2)
            z+=zr.permute(0, 2, 1)


        out = z / (self.n_rank*self.n_basis)

        if self.use_normalization:
            out = torch.sign(out) * (torch.sqrt(torch.abs(out)+1e-2)-np.sqrt(1e-2))
            # out = torch.sign(out) * torch.sqrt(torch.abs(out))
            # out = F.normalize(out, p=2, dim=1)
            out = self.channel_max_normalization(out)

        return out



class RPGaussianPooling(nn.Module):
    def __init__(self, 
                 n_basis=8, 
                 n_rank=4,
                 init_sigma = None,
                 use_normalization=False,
                 in_dim = 64):
        super(RPGaussianPooling, self).__init__()
        self.n_basis = n_basis
        self.n_rank = n_rank
        self.in_dim=in_dim
        self.init_sigma = init_sigma
        self.E_list = nn.ParameterList([])
        self.F_list = nn.ParameterList([])
        self.sigma_list = nn.ParameterList([])
        self.rho_list = nn.ParameterList([])
        self.use_normalization = use_normalization
        np.random.seed(seed=15385)

        if self.init_sigma is None:
            self.init_sigma = np.sqrt(self.in_dim)

        for r in range(self.n_rank):
            Er_np_G = np.random.standard_normal([self.in_dim, self.in_dim])
            Er_np_square,_ = np.linalg.qr(Er_np_G)
            Er_np = Er_np_square[:, :self.n_basis]
            sigma_r = Parameter(torch.tensor(self.init_sigma, 
                                   dtype=torch.float32, 
                                   device=torch.device('cuda:0')),
                                requires_grad=True)
            self.sigma_list.append(sigma_r)
            Er = Parameter(torch.tensor(Er_np,dtype=torch.float32,
                                        device=torch.device('cuda:0')), 
                           requires_grad=False)
            self.E_list.append(Er)
                               
            Fr_np_G = np.random.standard_normal([self.in_dim, self.in_dim])
            Fr_np_square,_ = np.linalg.qr(Fr_np_G)
            Fr_np = Fr_np_square[:, :self.n_basis]
            rho_r = Parameter(torch.tensor(self.init_sigma, 
                                   dtype=torch.float32, 
                                   device=torch.device('cuda:0')),
                                requires_grad=True)
            self.rho_list.append(rho_r)
            Fr = Parameter(torch.tensor(Fr_np,dtype=torch.float32,
                                        device=torch.device('cuda:0')), 
                           requires_grad=False)
            self.F_list.append(Fr)
             
                            
    def channel_max_normalization(self, x):
        max_vals = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
        return x / (max_vals+1e-5)


    def forward(self, x):
        """
        input feature x: [batch, feature_dim, time]
        """
        in_time = x.shape[2]

        z = 0
        for r in range(self.n_rank):

            Er = self.in_dim**0.5 / (1e-5+torch.abs(self.sigma_list[r])) * self.E_list[r] 
            Fr = self.in_dim**0.5 / (1e-5+torch.abs(self.rho_list[r])) * self.F_list[r] 
            
            #Er = self.E_list[r] 
            #Fr = self.F_list[r] 
         
            xer = torch.matmul(x.permute([0,2,1]), Er).unsqueeze(-1)
            xfr = torch.matmul(x.permute([0,2,1]), Fr).unsqueeze(-2)
            zr = torch.matmul(xer, xfr).view(-1, in_time, self.n_basis**2)
            z+=zr.permute(0, 2, 1)


        out = z / (float(self.n_rank))

        if self.use_normalization:
            out = torch.sign(out) * (torch.sqrt(torch.abs(out)+1e-2)-np.sqrt(1e-2))
            # out = torch.sign(out) * torch.sqrt(torch.abs(out))
            # out = F.normalize(out, p=2, dim=1)
            out = self.channel_max_normalization(out)

        return out




class RPGaussianPoolingFull(nn.Module):
    def __init__(self, 
                 n_basis=8, 
                 n_rank=4,
                 init_sigma = None,
                 use_normalization=False):
        super(RPGaussianPoolingFull, self).__init__()
        self.n_basis = n_basis
        self.n_rank = n_rank
        self.in_dim=64
        self.init_sigma = init_sigma
        self.E_list = nn.ParameterList([])
        self.F_list = nn.ParameterList([])
        self.sigma_list = nn.ParameterList([])
        self.rho_list = nn.ParameterList([])
        self.use_normalization = use_normalization
        np.random.seed(seed=1538574472)

        if self.init_sigma is None:
            self.init_sigma = np.sqrt(self.in_dim)


        for r in range(self.n_rank):
            Er_np_G = np.random.standard_normal([self.in_dim, self.in_dim])
            Er_np_square,_ = np.linalg.qr(Er_np_G)
            Er_np = Er_np_square[:, :self.n_basis//2]
            sigma_r = Parameter(torch.tensor(self.init_sigma, 
                                   dtype=torch.float32, 
                                   device=torch.device('cuda:0')),
                                requires_grad=True)
            self.sigma_list.append(sigma_r)
            Er = Parameter(torch.tensor(Er_np,dtype=torch.float32,
                                        device=torch.device('cuda:0')), 
                           requires_grad=False)
            self.E_list.append(Er)
                               
            Fr_np_G = np.random.standard_normal([self.in_dim, self.in_dim])
            Fr_np_square,_ = np.linalg.qr(Fr_np_G)
            Fr_np = Fr_np_square[:, :self.n_basis//2]
            rho_r = Parameter(torch.tensor(self.init_sigma, 
                                   dtype=torch.float32, 
                                   device=torch.device('cuda:0')),
                                requires_grad=True)
            self.rho_list.append(rho_r)
            Fr = Parameter(torch.tensor(Fr_np,dtype=torch.float32,
                                        device=torch.device('cuda:0')), 
                           requires_grad=False)
            self.F_list.append(Fr)
             
                            
    def channel_max_normalization(self, x):
        max_vals = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
        return x / (max_vals+1e-5)


    def forward(self, x):
        """
        input feature x: [batch, feature_dim, time]
        """
        in_time = x.shape[2]

        z = 0
        for r in range(self.n_rank):
            Er = self.in_dim**0.5 / (1e-5+torch.abs(self.sigma_list[r])) * self.E_list[r] 
            Fr = self.in_dim**0.5 / (1e-5+torch.abs(self.rho_list[r])) * self.F_list[r] 
            
            xer_cos = torch.cos(torch.matmul(x.permute([0,2,1]), Er).unsqueeze(-1))
            xer_sin = torch.sin(torch.matmul(x.permute([0,2,1]), Er).unsqueeze(-1))
            xer = torch.cat([xer_cos, xer_sin],dim=-2)

            xfr_cos = torch.cos(torch.matmul(x.permute([0,2,1]), Fr).unsqueeze(-2))
            xfr_sin = torch.sin(torch.matmul(x.permute([0,2,1]), Fr).unsqueeze(-2))
            xfr = torch.cat([xfr_cos, xfr_sin],dim=-1)

            zr = torch.matmul(xer, xfr).view(-1, in_time, self.n_basis**2)
            z+=zr.permute(0, 2, 1)


        out = z / float(self.n_rank)

        if self.use_normalization:
            out = torch.sign(out) * (torch.sqrt(torch.abs(out)+1e-2)-np.sqrt(1e-2))
            # out = torch.sign(out) * torch.sqrt(torch.abs(out))
            # out = F.normalize(out, p=2, dim=1)
            out = self.channel_max_normalization(out)

        return out




class RPLearnablePooling(nn.Module):
    def __init__(self, 
                 n_basis=8, 
                 n_rank=1,
                 use_normalization=False):
        super(RPLearnablePooling, self).__init__()
        self.n_basis = n_basis
        self.n_rank = n_rank
        self.E_list = []
        self.F_list = []
        self.in_dim=64
        for r in range(self.n_rank):
            Er = np.random.standard_normal([self.in_dim, self.n_basis])
            self.E_list.append(Parameter(torch.tensor(Er,dtype=torch.float32,
                                                      device=torch.device('cuda:0')), 
                                        requires_grad=True)
                               )
            Fr = np.random.standard_normal([self.in_dim, self.n_basis])
            self.F_list.append(Parameter(torch.tensor(Fr,dtype=torch.float32,
                                                      device=torch.device('cuda:0')), 
                                        requires_grad=True)
                               )

    def forward(self, x):
        in_time = x.shape[2]

        z = 0
        for r in range(self.n_rank):
            
            xer = torch.matmul(x.permute([0,2,1]), self.E_list[r]).unsqueeze(-1)
            xfr = torch.matmul(x.permute([0,2,1]), self.F_list[r]).unsqueeze(-2)
            zr = torch.matmul(xer, xfr).view(-1, in_time, self.n_basis**2)
            z += zr.permute(0, 2, 1)

        return z / float(self.n_rank)







class MLBPooling(nn.Module):
    def __init__(self, 
                 n_basis=8,
                 use_normalization=False):
        super(MLBPooling, self).__init__()
        self.n_basis = n_basis
        self.in_dim=64
        self.use_normalization = use_normalization


        Er = np.random.standard_normal([self.in_dim, self.n_basis])
        self.Er_torch=Parameter(torch.tensor(Er,dtype=torch.float32,
                                                  device=torch.device('cuda:0')), 
                                    requires_grad=True)

        Fr = np.random.standard_normal([self.in_dim, self.n_basis])
        self.Fr_torch=Parameter(torch.tensor(Fr,dtype=torch.float32,
                                                device=torch.device('cuda:0')), 
                                    requires_grad=True)
        
        self.b = Parameter(torch.zeros([1, self.n_basis, 1], 
                                       dtype=torch.float32,
                                       device=torch.device('cuda:0')),
                           requires_grad=True)         
    

                                   
    def channel_max_normalization(self, x):
        max_vals = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
        return x / (max_vals+1e-5)




    def forward(self, x):
        in_time = x.shape[2]
        
        xer = torch.matmul(x.permute([0,2,1]), self.Er_torch).permute([0,2,1])
        xfr = torch.matmul(x.permute([0,2,1]), self.Fr_torch).permute([0,2,1])
        out = xer * xfr + self.b

                   
        if self.use_normalization:
            out = torch.sign(out) * (torch.sqrt(torch.abs(out)+1e-2)-np.sqrt(1e-2))
            # out = torch.sign(out) * torch.sqrt(torch.abs(out))
            # out = F.normalize(out, p=2, dim=1)
            out = self.channel_max_normalization(out)


        return out



class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes,
                 pooling_type, dropout):
        super(MultiStageModel, self).__init__()

        if pooling_type in ['RPGaussian', 'RPBinary', 'RPGaussianFull', 'Hadamard', 'FBM']:

            self.stage1 = SingleStageModelBilinear(num_layers, num_f_maps, dim, num_classes,
                                            pooling_type=pooling_type, 
                                            dropout=dropout)
            self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, 
                                                num_f_maps, num_classes, num_classes)) 
                                        for s in range(num_stages-1)])

        else:
            self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
            self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, 
                                                num_f_maps, num_classes, num_classes)) 
                                        for s in range(num_stages-1)])


    def forward(self, x):
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs



class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out



class SingleStageModelBilinear(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes,
        pooling_type, dropout):
        super(SingleStageModelBilinear, self).__init__()
        sqrt_dim = int(np.sqrt(num_f_maps))
        # sqrt_dim = np.sqrt(dim)
        dim_factor = 2
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1, padding=0)
        self.conv_1x1_b = nn.Conv1d(dim_factor*dim_factor*sqrt_dim**2, num_classes, 25, padding=12)
        self.drop = nn.Dropout(p=dropout)
        self.drop_p = dropout

        if pooling_type=='RPBinary':
            self.bilinear_layer = RPBinaryPooling(n_basis=int(dim_factor*sqrt_dim), 
                                                 n_rank=4, 
                                                 use_normalization=False)
        elif pooling_type=='RPGaussian':
            self.bilinear_layer = RPGaussianPooling(n_basis=int(dim_factor*sqrt_dim), 
                                                   n_rank=4, 
                                                   init_sigma=sqrt_dim,
                                                   use_normalization=False)
        elif pooling_type == 'RPGaussianFull':
            self.bilinear_layer = RPGaussianPoolingFull(n_basis=int(dim_factor*sqrt_dim), 
                                                   n_rank=4, 
                                                   init_sigma=sqrt_dim,
                                                   use_normalization=False)
        elif pooling_type=='Hadamard':
            self.bilinear_layer = MLBPooling(n_basis=dim_factor**2 *num_f_maps,
                                             use_normalization=False)
        elif pooling_type=='FBM':
            self.bilinear_layer = FBM(n_basis=dim_factor**2 *num_f_maps,
                                      use_normalization=False)
        else:
            print('[Error]: no such bilinear layer. Program terminates')
            sys.exit()


    def forward(self, x):

        out = self.conv_1x1(x)

        for layer in self.layers:
            out = layer(out)

        # # ####### apply bilinear residual module here! ####
        out2 = self.bilinear_layer(out)
        out2 = self.conv_1x1_b(out2)
        if self.drop_p > 0:
            out2 = self.drop(out2)
        
        # #########################################
        #out = self.conv_out(out)

        return out2



class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out



class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes,
        pooling_type, dropout):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes,
            pooling_type=pooling_type, dropout=dropout)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        #print('[INFO] -----------device: '+ str(torch.cuda.get_device_name()))


    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        loss_lambda = 0.15
        loss_tau = 4.0
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):

            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += loss_lambda*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=loss_tau**2)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            acc = float(correct)/total
            # if acc-acc_pre < -0.005:
            #     break
            if (epoch+1) % 10 == 0:
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                                float(correct)/total))



    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            # epoch=50
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                #print (vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    action_dict_key_list = list(actions_dict.keys())
                    recognition = np.concatenate((recognition, [action_dict_key_list[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
