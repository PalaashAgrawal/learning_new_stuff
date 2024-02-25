#TODO: parallelization of model weights across GPUs for larger models. 

import torch
import torch.nn as nn
import torch.nn.functional as F


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
n_head = 6 #ie head_size = 384/6 = 64
n_layer = 6
# ------------

class charTokenizer():
    def __init__(self, input_file_path):
        with open(input_file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
    
        chars =sorted(list(set(self.text)))
        self.vocab_size = len(chars)
    
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}
    
    def encode(self, s):
        f'takes a string, returns list of integers'
        return [self.stoi[c] for c in s] #takes a string, ouputs a list of integers
    
    def decode(self, l):
        f'takes a list of token numbers, converts to string'
        return ''.join([self.itos[i] for i in l]) #
    


tokenizer = charTokenizer('input.txt')
vocab_size = tokenizer.vocab_size



#what if we use actual GPT3.5 model tokenizer?

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
vocab_size = enc.n_vocab
print('vocab_size: ', vocab_size)





#____________________________________________________________________________________________________________________________________________



class Head(nn.Module):

    """one head of the self-attention"""

    def __init__(self, head_size):
    
        super().__init__()
        self.key = nn.Linear(n_embd, head_size) #by default initialization, variance is 1. 
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #to constrain flow of info from future to current tokens. Important in autoregressive systems. 
        

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
                                 nn.Linear(n_embd*4, n_embd),
                                 nn.Dropout(dropout),
                                 )
        
    
    def forward(self, x): return self.net(x)


class Block(nn.Module):
    """ an implementation of the transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        #n_embd: embedding dimension. n_head, number of self attention heads we'd like. 
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.f = FeedForward(n_embd)

        self.layernorm1 = nn.LayerNorm(n_embd) #the layer referred to as Add & Norm in the original transformer block diagram
        self.layernorm2 = nn.LayerNorm(n_embd)

        

    def forward(self, x):
        """
        Note:
        Most architectural design choices from the original transformers paper havent changed since the paper came out. 
        But a few changes have been changed in practice, because they have shown better results. 
        One of such changes is the placement of the layernorm (Add&Norm) layer in the transformer block. 
        Instead of AFTER the multihead attention and feedforward layer, the layernorm, in practice, is now usually added BEFORE these layers. This is called the PreNormalization. 

        Source: Original GPT2 paper (language models are unsupervised multitask learners) mentions, (which was also used by the GPT3 paper (Language models are few shot learners):
           (section 2.3 of GPT2 paper) "The model largely follows ...with a few modifications. Layer Norm was moved to the input of each sub-block, similar to pre-activation residual network " 
        
        """
        x = x+ self.sa(self.layernorm1(x)) #in the tfm block, you see one residual connection in the multihead attention part, 
        x = x+ self.f(self.layernorm2(x)) #and another residual connection in the the feedforward part
        return x




class GPTLanguageModel(nn.Module):
    name = 'GPT'
    def __init__(self):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm: 
        #According to original GPT2 paper "Language Models are Unsupervised Multitask Learners", section 2.3: an additional layer normalization was added after the final self-attention block. 

        #final linear layer to project embeddings back to one hot encoded vectors for decoding.
        self.lm_head = nn.Linear(n_embd, vocab_size)

        #better initialization for transformers. Weights are normally initialized, bias is initialized to zeros
        self.apply(self._init_weights)
    
    
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std =0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean =0.0, std =0.02) #this initialization also comes from GPT2 paper. But how does 0.02 come?
    

    def forward(self, idx, targets =None):
        B,T = idx.shape
        
        #idx and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(idx) #(B,T,C)
        pos_emb = self.position_embedding_table(idx) #(B,T,C)

        x = tok_emb + pos_emb #(B,T,C)
        x = self.blocks(x) #(B,T,C)
        x = self.ln_f(x) #(B,T,C)
        logits = self.lm_head(x) #(B,T,C)

        if targets is None: loss=None
        else: 
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens = 200):
        f"""idx is (B,T) array of indices in the current context. 
        this function auto-regressively continues character prediction. 

        We take out one prediction 
        """

        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] #ie, the latest blocksize or less tokens will be taken at once. Rest rejected.
            #get preds
            logits, loss = self(idx_cond)
            #lets take only the last token, since thats the newly generated token, and we want to print it
            logits = logits[:, -1, :] #becomes (B,C) --> pytorch does not retain the dimension for sigle value tensor slicing. 
            #apply softmax to get probabilities. 
            probs = F.softmax(logits, dim = -1) # (B,C)

            #sample from the distribution: Instead of just choosing the token with maximum logit. This is done to avoid repetition of the same output everytime, and introduce variability. 
            #this makes sense for large token embedding spaces, where there are many tokens with similar (and high) logit values. 

            idx_next = torch.multinomial(probs, num_samples = 1) #(B,1)
            #append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) #(B,T+1)
        
        return idx

    



    def __repr__(self): 
        return f'{self.name} Model architecture with {sum(p.numel() for p in self.parameters())/1e6}M parameters.'



model = GPTLanguageModel()
m = model.to(device)
print(m.__repr__)
