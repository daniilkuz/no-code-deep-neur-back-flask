import torch
import torch.nn.functional as F
# import torch.nn as nn
import random
from utils import generateId
import pickle

# g = torch.Generator().manual_seed(2147483647) # for reproducibility
        
# ix=None
# stoi = []
# C = None
# Xtr, Ytr, Xval, Yval, Xtest, Ytest = None, None, None, None, None, None
# layers = []
# parameters=None
# lossi = []
# ud = []

class Linear:
    
    def __init__(self, fan_in, fan_out, generator=None, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=generator) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):
        # calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True) # batch mean
            xvar = x.var(0, keepdim=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
  
  def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []

class Model:
    def __init__(self, words=None, C=None, itos=None, layers=None, block_size=3, batch_size=30, n_embd=10, max_steps = 1000):
        self.g = torch.Generator().manual_seed(2147483647)
        self.words = words
        self.stoi = None
        self.itos = itos
        self.block_size = block_size
        self.batch_size = batch_size
        self.n_embd = n_embd
        self.max_steps = max_steps
        self.C = C
        self.Xtr, self.Ytr, self.Xval, self.Yval, self.Xtest, self.Ytest = None, None, None, None, None, None
        self.layers = layers
        self.parameters=None
        self.lossi = []
        self.ud = []

    def _build_dataset(self, words):
        X, Y = [], []
        for w in words:
            context = [0]*self.block_size
            # print("lkj2")
            # print(self.stoi)
            for ch in w + ".":
                # print("lkj1")
                ix = self.stoi[ch]
                # print("lkj")
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
                # print("lkj3")

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        print(X.shape, Y.shape)
        return X, Y

    @torch.no_grad()
    def split_loss(self, split):
        x, y = {'train': (self.Xtr, self.Ytr), 'val': (self.Xval, self.Yval), 'test': (self.Xtest, self.Ytest)}[split]
        embd = self.C[x]
        x = embd.view(embd.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        loss = F.cross_entropy(x, y)
        print(loss, loss.item())
        return loss.item()

    def _initialize(self, neurons_per_layer=None, n_hidden = 100):
        # words = open("names.txt", "r").read().splitlines()
        
        print("I am here")
        chars = sorted(list(set(''.join(self.words))))
        print(chars)
        self.stoi = {s: i+1 for i, s in enumerate(chars)}
        self.stoi["."] = 0
        self.itos = {i: s for s, i in self.stoi.items()}
        vocab_size = len(self.itos)

        print(self.block_size, self.n_embd)
        random.seed(42)
        random.shuffle(self.words)
        print('stoi: ', self.stoi)
        n1 = int(0.8 * len(self.words))
        n2 = int(0.9 * len(self.words))
        print("after vocab_sizes")
        self.Xtr, self.Ytr = self._build_dataset(self.words[:n1])
        self.Xval, self.Yval = self._build_dataset(self.words[n1:n2])
        self.Xtest, self.Ytest = self._build_dataset(self.words[n2:])
        print("after 3")
        # Let's train a deeper network
        self.C = torch.randn((vocab_size, self.n_embd), generator=self.g)
        print('layers before: ', neurons_per_layer)
        if neurons_per_layer:
            self.layers = []
            neuronsBefore = self.n_embd * self.block_size
            for neurons in neurons_per_layer:
                self.layers.append(Linear(neuronsBefore, neurons, bias=False))
                self.layers.append(BatchNorm1d(neurons))
                self.layers.append(Tanh())
                neuronsBefore=neurons
            self.layers.append(Linear(neuronsBefore, vocab_size, bias=False))
            self.layers.append(BatchNorm1d(vocab_size))
        else:
            self.layers = [
                Linear(self.n_embd * self.block_size, n_hidden, bias=False, generator=self.g), BatchNorm1d(n_hidden), Tanh(),
                Linear(           n_hidden, n_hidden, bias=False, generator=self.g), BatchNorm1d(n_hidden), Tanh(),
                Linear(           n_hidden, n_hidden, bias=False, generator=self.g), BatchNorm1d(n_hidden), Tanh(),
                Linear(           n_hidden, n_hidden, bias=False, generator=self.g), BatchNorm1d(n_hidden), Tanh(),
                Linear(           n_hidden, n_hidden, bias=False, generator=self.g), BatchNorm1d(n_hidden), Tanh(),
                Linear(           n_hidden, vocab_size, bias=False, generator=self.g), BatchNorm1d(vocab_size),
                ]
        print("layers after: ", self.layers)

        with torch.no_grad():
            # last layer: make less confident
            self.layers[-1].gamma *= 0.1
            #layers[-1].weight *= 0.1
            # all other layers: apply gain
            for layer in self.layers[:-1]:
                if isinstance(layer, Linear):
                    layer.weight *= 1.0 #5/3
        
        self.parameters = [self.C] + [p for layer in self.layers for p in layer.parameters()]
        print(sum(p.nelement() for p in self.parameters)) # number of parameters in total
        for p in self.parameters:
            p.requires_grad = True

    def neural_network_train(self, words, block_size, neurons_per_layer=None, n_embd = 10, n_hidden = 100,max_steps = 1000, batch_size = 32):
        # self.g = torch.Generator().manual_seed(2147483647) # for reproducibility
        print('params: ', block_size, n_embd, max_steps, batch_size)
        if words!=None:
            self.words = words
        if block_size!=None:
            self.block_size = block_size
        if n_embd!=None:
            self.n_embd = n_embd
        if batch_size!=None:
            self.batch_size = batch_size
        if max_steps!=None:
            self.max_steps = max_steps
        self._initialize(neurons_per_layer=neurons_per_layer, n_hidden = n_hidden)
        print('after init')
        for i in range(self.max_steps):
            # minibatch construct
            ix = torch.randint(0, self.Xtr.shape[0], (self.batch_size,), generator=self.g)
            Xb, Yb = self.Xtr[ix], self.Ytr[ix] # batch X,Y
            # forward pass
            emb = self.C[Xb] # embed the characters into vectors
            x = emb.view(emb.shape[0], -1) # concatenate the vectors
            for layer in self.layers:
                x = layer(x)
            loss = F.cross_entropy(x, Yb) # loss function
        
            # backward pass
            for layer in self.layers:
                layer.out.retain_grad() # AFTER_DEBUG: would take out retain_graph
            for p in self.parameters:
                p.grad = None
            loss.backward()

            # update
            lr = 0.1 if i < 150000 else 0.01 # step learning rate decay
            for p in self.parameters:
                p.data += -lr * p.grad

            # track stats
            if i % 10000 == 0: # print every once in a while
                print(f'{i:7d}/{self.max_steps:7d}: {loss.item():.4f}')
            self.lossi.append(loss.log10().item())
            with torch.no_grad():
                self.ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in self.parameters])

            # if i >= 1000:
            #   break # AFTER_DEBUG: would take out obviously to run full optimization

        for layer in self.layers:
            layer.training = False
        train_loss = self.split_loss('train')
        val_loss = self.split_loss('val')
        # return self.C, self.layers, self.itos, train_loss, val_loss
        return {"train_loss": train_loss, "val_loss": val_loss, 
                "batch_size": self.batch_size, "block_size": self.block_size, 
                "max_steps": self.max_steps, "n_embd": self.n_embd,
                "neurons_per_layer": neurons_per_layer}

    # sample from the model

    def sample_from_model(self, n_samples=50):
        self.g = torch.Generator().manual_seed(2147483647 + 10)
        result = ''
        for _ in range(n_samples):
            out = []
            context = [0] * self.block_size # initialize with all ...
            while True:
                # forward pass the neural net
                emb = self.C[torch.tensor([context])] # (1,block_size,n_embd)
                x = emb.view(emb.shape[0], -1) # concatenate the vectors
                for layer in self.layers:
                    x = layer(x)
                logits = x
                probs = F.softmax(logits, dim=1)
                # sample from the distribution
                ix = torch.multinomial(probs, num_samples=1, generator=self.g).item()
                # shift the context window and track the samples
                context = context[1:] + [ix]
                out.append(ix)
                # if we sample the special '.' token, break
                if ix == 0:
                    break
            print(''.join(self.itos[i] for i in out))
            result = result + ''.join(self.itos[i] for i in out) + '\n'  
        return result
    
    def save_model(self):
        model_name = "files/model" + generateId()+".pickle"
        with open(model_name, "wb") as f:
            pickle.dump({"C": self.C, "layers": self.layers, "itos": self.itos, "block_size": self.block_size}, f)
        print('saved!')
        return model_name
