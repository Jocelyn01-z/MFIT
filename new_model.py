import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.init as init


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

mish_layer = Mish()
class encoder(nn.Module):
    def __init__(self, n_dim, dims, n_z, dropout_prob=0.2):
        super(encoder, self).__init__()
        self.enc_layers = nn.ModuleList()
        self.enc_layers.append(nn.Sequential(
            nn.Linear(n_dim, dims[0]),
            nn.BatchNorm1d(dims[0]),
            Mish(),
            # nn.ReLU(),
            nn.Dropout(dropout_prob)
        ))
        self.enc_layers.append(nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.BatchNorm1d(dims[1]),
            Mish(),
            # nn.ReLU(),
            nn.Dropout(dropout_prob)
        ))
        self.enc_layers.append(nn.Sequential(
            nn.Linear(dims[1], dims[2]),
            nn.BatchNorm1d(dims[2]),
            Mish(),
            # nn.ReLU(),
            nn.Dropout(dropout_prob)
        ))
        self.z_layer = nn.Linear(dims[2], n_z)
        self.z_b0 = nn.BatchNorm1d(n_z)

    def forward(self, x):
        for layer in self.enc_layers:
            x = layer(x)
        z = self.z_b0(self.z_layer(x))
        return z



class net(nn.Module):

    def __init__(self, n_stacks, n_input, n_z, nLabel):
        super(net, self).__init__()


        dims = []
        for n_dim in n_input:

            linshidims = []
            for idim in range(n_stacks - 2):
                linshidim = round(n_dim * 0.8)
                linshidim = int(linshidim)
                linshidims.append(linshidim)
            linshidims.append(1500)
            dims.append(linshidims)#[80,80,1500]

        self.encoder_list = nn.ModuleList([encoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])

  
        self.act = nn.Sigmoid()
        self.nLabel = nLabel
        self.BN = nn.BatchNorm1d(n_z)
        self.nz=n_z
        self.num_views=len(n_input)
        
        self.fc1 = torch.nn.Linear(n_z*2, nLabel*2)
    

    def DS_Combin_two(self, alpha1_tensor, alpha2_tensor):
        alpha = dict()
        c=alpha1_tensor.shape[1]
        alpha[0], alpha[1] = alpha1_tensor,alpha2_tensor
       
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=0,keepdim=True)#(1,c)
            E[v] = alpha[v] - 1#(2,c)
            b[v] = E[v] / (S[v].expand(E[v].shape))#(2,c)
            u[v] = 2 / S[v]#(1,c)
        
        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].T.view(c, 2, 1), b[1].T.view(c, 1, 2))#[c, 2, 2]
        
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand) #(2,c)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand) #(2,c)
        
       
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)# (c,)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)#(c,)
        
        K = bb_sum - bb_diag#(c,)
        
        
        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) /((1 - K).unsqueeze(0).expand(2, -1) )# (2,c)
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - K).unsqueeze(0))#(1,c)
        
       
        S_a = 2 / u_a#(1,c)
   
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))#(2,c)
        alpha_a = e_a + 1#(2,c)
        return alpha_a
    
    def combine_views(self, alpha_list, W):
        n = W.size(0)  
        c = alpha_list[0][0].size(1)
        
     
        combined_alpha = torch.zeros((n,2, c), 
                                    device=alpha_list[0][0].device,  
                                    dtype=alpha_list[0][0].dtype)   
        
        for i in range(n):
            existing_views = [j for j in range(W.shape[1]) if W[i, j].item() == 1]
            
            if not existing_views:
                
                continue
            elif len(existing_views) == 1:
                j = existing_views[0]
                combined_alpha[i] = torch.stack([alpha_list[k][j][i] for k in (0,1)], dim=0)
            else:
                
                current =torch.stack([alpha_list[k][existing_views[0]][i] for k in (0,1)], dim=0)#(2,c)
                for j in existing_views[1:]:
                    current = self.DS_Combin_two(current, torch.stack([alpha_list[k][j][i] for k in (0,1)], dim=0))
                combined_alpha[i] = current#(2,c)
       
        return combined_alpha#(n,2,c)

    def DSTdynamic(self, alpha_list, W,mass):
       
        
        n = W.size(0)  
        c = alpha_list[0][0].size(1)
        combined_alpha = torch.zeros((n,2, c), 
                                    device=alpha_list[0][0].device,  
                                    dtype=alpha_list[0][0].dtype)     
        
        
        alpha_pos_tensor = torch.stack(alpha_list[0],dim=1)# (n,6,c)
        alpha_neg_tensor = torch.stack(alpha_list[1],dim=1)#(n,6,c)
        mass = mass.unsqueeze(2)
        WAE_pos = torch.sum(alpha_pos_tensor*mass,dim=1)#(n,c)
        WAE_neg = torch.sum(alpha_neg_tensor*mass,dim=1)
        
        WAE = torch.stack([WAE_pos,WAE_neg],dim=1)#(n,2,c)
        
        for i in range(n):
            for v in range(W.shape[1]):
                if v==0:
                    alpha_a = self.DS_Combin_two(WAE[i], WAE[i])
                else:
                    alpha_a = self.DS_Combin_two(alpha_a, WAE[i])
            combined_alpha[i] = alpha_a
       
        return combined_alpha
    
    def compute_sample_gram(self,mask_row, view_predictions):
        
        existing_views = mask_row.nonzero().squeeze(dim=1)  
        existing_vectors = [view_predictions[vid] for vid in existing_views]  
        vectors_stacked = torch.stack(existing_vectors)  
        gram_matrix = vectors_stacked @ vectors_stacked.T 
        return gram_matrix
    
    def corrected_disagreement_from_gram(self,gram_matrix):
       
        k = gram_matrix.shape[0]
        
        if k == 1:
            return torch.tensor(0.0)
        
        diag = torch.diag(gram_matrix)
        if torch.any(diag < 1e-8):
            return torch.tensor(1.0)
        
        disagreement_terms = []
        specifi_d = torch.zeros(1, k)
        for i in range(k):
            for j in range(i+1, k):
                dot_product = gram_matrix[i][j]
                norm_product = torch.sqrt(diag[i] * diag[j])
                cos_theta = dot_product / norm_product
                cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
                disagreement = (1 - cos_theta) / 2
                disagreement_terms.append(disagreement)
            specifi_d[:,i] = torch.mean(torch.tensor(disagreement_terms)) 
        
       
        return specifi_d
    
    def batch_disagreement_score(self,mask, A):
        """Batch calculate the divergence degree of all samples"""
        n = mask.shape[0]
        scores = []
        result = torch.zeros_like(mask, dtype=torch.float32)
        
        for i in range(n):
            mask_row = mask[i]
            sample_predictions = [A[vid][i] for vid in range(mask.shape[1])]
            gram = self.compute_sample_gram(mask_row, sample_predictions)
            score = self.corrected_disagreement_from_gram(gram)
            mass = 1 - score
            mass = mass+1
            ones_positions = mask_row.nonzero().squeeze(dim=1) 
            mass_squeezed = mass.squeeze(0)
            result[i, ones_positions] = mass_squeezed.to(mask_row.device) 
        return result
    
    def transform_to_frequency_domain(self,features):
        n, d = features.shape
        fft_features = torch.fft.fft(features, dim=1)
        freq = torch.fft.fftfreq(d)[:d // 2].cpu().numpy()  
        fft_magnitude = torch.abs(fft_features)
        return fft_magnitude, freq

    def forward(self, mul_X, we,G,mode):
        
        share_zs = []
        mass_list=[]
        evidence_alpha_list=[]
        evidence_beta_list=[]
        alpha_pos_list = []
        alpha_neg_list = []
        alpha_list = []

        if mode =='train':
            for i,X in enumerate(mul_X):
                mask_len = int(0.25*X.size(-1))
                mask = torch.ones_like(X)
                for j in range(mask.shape[0]):
                    zero_indices = torch.randperm(mask.shape[1])[:mask_len]
                    mask[j, zero_indices] = 0
                mul_X[i] = mul_X[i].mul(mask)

        for enc_i, enc in enumerate(self.encoder_list):        

            z_i = enc(mul_X[enc_i])
            share_zs.append(z_i)
            
        # frequency feature
        fft_views = []
        freqs = None
        for view in share_zs:
            fft_view, freq = self.transform_to_frequency_domain(view)
            fft_views.append(fft_view) 
            if freqs is None:
                freqs = freq
        # Z-score_Norm
        concat_zs = []
        for i,freq_feat in enumerate(fft_views):
            freq_mean = freq_feat.mean(dim=0, keepdim=True)
            freq_std = freq_feat.std(dim=0, keepdim=True) + 1e-8
            freq_feat_norm = (freq_feat - freq_mean) / freq_std
            
            concat_feat = torch.cat([share_zs[i], freq_feat_norm], dim=1)
            concat_zs.append(concat_feat)
        
        # Evidential Deep Learning 
        for i in range(we.shape[1]):
           
            out = self.fc1(concat_zs[i]) #n_z->class_num*2  
            out = F.softplus(out)
            
            evidence_alpha, evidence_beta = torch.split(out, self.nLabel, 1)

            evidence_alpha_list.append(evidence_alpha)
            evidence_beta_list.append(evidence_beta)
            
            
            alpha_pos = evidence_alpha +1
            alpha_neg = evidence_beta +1
            
            alpha_pos_list.append(alpha_pos)#(n,c)
            alpha_neg_list.append(alpha_neg)
        
        stacked_alpha = [torch.cat([pos.unsqueeze(1) , neg.unsqueeze(1) ], dim=1)  # (n,2,c)
           for pos, neg in zip(alpha_pos_list, alpha_neg_list)]
       
        # fusion
        specif_B = []
        for i in range(we.shape[1]):
            specif_B.append((stacked_alpha[i][:,0,:].squeeze(1))/torch.sum(stacked_alpha[i],dim=1))#(n,c)
        mass = self.batch_disagreement_score(we, specif_B)#(n,view_num)
      
        for i in range(we.shape[1]):
            alpha_pos_list[i]=alpha_pos_list[i]*mass[:,i].unsqueeze(1).to(alpha_pos_list[i].device)
            alpha_neg_list[i]=alpha_neg_list[i]*mass[:,i].unsqueeze(1).to(alpha_neg_list[i].device)
        alpha_list.append(alpha_pos_list)
        alpha_list.append(alpha_neg_list)
        fusion_alpha = self.combine_views(alpha_list, we)#(n,2,c)
        
        return stacked_alpha,fusion_alpha

class AsymmetricBetaLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.1, k=1):
        super(AsymmetricBetaLoss, self).__init__()
        self.k = k
        self.clip = clip  
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
          
    def forward(self, B_alpha, B_beta, y):
       
        Lp = torch.digamma(B_alpha + B_beta + self.gamma_pos) - torch.digamma(B_alpha)
        
        m = torch.tensor(self.clip)  
        B_alpha_m = torch.max(B_alpha - m / (1 - m) * B_beta, torch.zeros_like(B_alpha))

        Ln = torch.digamma(B_alpha_m + B_beta + self.gamma_neg) - torch.digamma(B_beta)

        torch.set_grad_enabled(False)
        w_pos = torch.ones_like(B_alpha, dtype=float)
        w_neg = torch.ones_like(B_beta, dtype=float)
        for i in range(self.gamma_pos):
            w_pos *= (B_beta + i) / (B_alpha + B_beta + i)
        for i in range(self.gamma_neg):
            w_neg *= (B_alpha_m + i) / (B_alpha_m + B_beta + i)
        torch.set_grad_enabled(True)

        pos_loss = (w_pos * torch.pow(y, self.k) * Lp)
        neg_loss = (w_neg * torch.pow(1 - y, self.k) * Ln)

        return pos_loss + neg_loss    


def get_model(n_stacks,n_input,n_z,Nlabel,device):
    model = net(n_stacks=n_stacks,n_input=n_input,n_z=n_z,nLabel=Nlabel).to(device)
    return model
