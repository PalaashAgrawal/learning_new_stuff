import torch
import torch.nn as nn
import torch.nn.Functional as F


# hyperparameters set 1: data details. 
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?

#hyperparameters set 2: training
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200


#hyperparameters set 2: architecture details
n_embd = 384 #which is the vocabulary size
n_head = 6
n_layer = 6
# ------------



class Head(nn.Module):

    """one head of the self-attention"""

    def __init__(self, head_size):
    
        super().__init__()
        self.key = nn.Linear(n_embd, head_size) #by default initialization, variance is 1. 
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size), block_size)) #to constrain flow of info from future to current tokens. Important in autoregressive systems. 
        

        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        #x.shape = (batch, time-step, channels)
        #output.shape = (batch, time-step, hs), hs ~= head_size
        
        B,T,C = x.shape
        k = self.key(x) #B,T,hs
        q = self.query(x) #B,T,hs

        #compute attention scores ~= affinites
        w = q@k.transpore(-2,-1) * k.shape[-1]**-0.5 #B,T,T
        #normalizing with root of head_size value. Helps preserve variance to 1. 
        #Simply multiplying matrixes will give variance ~head_size (because elements in each row are summed accross columns, which are of dimension head_size in this case)

        w = w.masked_fill(self.tril[:T,:T] ==0, float('-inf')) # tril[:T, :T] in case x has less number of words than block_size. 0's are replaced by -infs, so that softmax automatically turns them into 0. 
        w = F.softmax(w, dim = -1) #B,T,T
        w = self.dropout(w)

        #perform the weighted aggregation of the values
        v = self.value(x) #B,T,hs
        out = w@v #(B,T,hs)
        
        return out
    




class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout) #to be applied on proj layer. There is already a dropout in each head

    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out  = self.dropout(self.proj(out))
        return out 


    

class FeedForward(nn.Module):
    """a simple linear layer + RELU that retains the shape of the input
        If you read section 3.3 from Attention is all you need: 
        'each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position 
        separately and identically. This consists of two linear transformations with a ReLU activation in between.
        ...The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality dff = 2048.'
    """
    def __init__(self, n_embd):
        super().__init__()
    
        self.net = nn.Sequential(nn.Linear(n_embd, n_embd*4), 
                                 nn.ReLU(),
                                 nn.Linear(n_embd*4, n_embd))
        
    
    def forward(self, x): return self.net(x)


class Block(nn.Module):
    """ an implementation of the transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        #n_embd: embedding dimension. n_head, number of self attention heads we'd like. 
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        #here, we're 
        self.f = FeedForward(n_embd)

        self.layernorm1 = nn.LayerNorm(n_embd) #the layer referred to as Add & Norm in the original transformer block diagram
        self.layernorm2 = nn.LayerNorm(n_embd)

        

    def forward(self, x):
        """
        Note:
        Most architectural design choices from the original transformers paper havent changed since the paper came out. 
        But a few changes have been changed in practice, because they have shown better results. 
        One of such changes is the placement of the layernorm (Add&Norm) layer in the transformer block. 
        Instead of AFTER the multihead attention and feedforward layer, the layernorm, in practice, is now usually added BEFORE these layers. 
        This is called the PreNormalization. Most recent papers (including ViT paper, for example), now use prenorm, instead of postnorm. 
        """
        x = x+ self.sa(self.layernorm1(x)) #in the tfm block, you see one residual connection in the multihead attention part, 
        x = x+ self.f(self.layernorm2(x)) #and another residual connection in the the feedforward part
        return x





