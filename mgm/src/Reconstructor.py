
import torch
from torch import nn
import math
import torch.optim as optim
from pytorch_lightning import LightningModule
from mgm.src.utils import loss_bc

class PositionEmbedding:
    def __init__(self, d_model=1, max_len=512, device='cpu'):
        self.d_model = d_model
        self.max_len = max_len
        self.device= device
        
        position = torch.arange(max_len, dtype = torch.float32, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * 
                        (-math.log(10000.0) / d_model))
        self.position_encoding = torch.zeros(max_len, d_model, device=device)
        self.position_encoding[:,0::2] = torch.sin(position * div)
        self.position_encoding[:,1::2] = torch.cos(position * div)
        
    def __call__(self,x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        batch_size, sentence_len = x.shape
        position_ids = torch.arange(sentence_len, dtype = torch.int64, 
                                    device=self.device).unsqueeze(0).repeat(batch_size, 1)
        return self.position_encoding[position_ids]
    

class reconstructorNet(LightningModule):
    def __init__(self, N, lr):
        super().__init__()
        M = 2 * N
        self.f1 = nn.Linear(N,M)
        self.f2 = nn.Linear(M,N)
        self.f3 = nn.Linear(N,N)
        self.lr = lr
    
    def forward(self, y):
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        mask = (y!=0).float()
        out = self.f1(y)
        out = torch.nn.ReLU()(out)
        out = self.f2(out)
        out = torch.nn.ReLU()(y + out)
        out = self.f3(out)
        out = torch.nn.Softmax()(y + out)
        return mask * out
    
    def predict(self, p):
        p_pred = self(p)
        return p_pred
    
    def training_step(self, batch, batch_idx):
        p, q = batch
        p_pred = self(p)
        loss = loss_bc(p_pred, q)
        self.log('train_loss', loss)  
        return loss

    def validation_step(self, batch, batch_idx):
        p, q = batch
        p_pred = self(p)  
        loss = loss_bc(p_pred, q)
        self.log('val_loss', loss)  
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
