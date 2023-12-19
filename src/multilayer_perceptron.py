#A multilayer perceptron
from typing import Any
import torch
import torch.nn.functional as F
import random
import time

namelist = open('./src/names.txt', 'r').read().splitlines()
ab = torch.zeros((27,27), dtype=torch.int32)
chars = sorted(list(set(''.join(namelist))))
charToInt = {s: i + 1 for i,s in enumerate(chars)}
charToInt['.'] = 0
intToChar = {i: s for s,i in charToInt.items()}
blockSize = 3
vocabSize = len(charToInt)
def buildDataSet(words):
  X, Y = [], []
  for w in words:
    context = [0] * blockSize
    for char in w + '.':
      ix = charToInt[char]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix]
  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X, Y

random.seed(42)
random.shuffle(namelist)
n1 = int(0.8*len(namelist))
n2 = int(0.9*len(namelist))

xTrain, yTrain = buildDataSet(namelist[:n1])
xDev, yDev = buildDataSet(namelist[n1:n2])
xTest, yTest = buildDataSet(namelist[n2:])

class Linear:
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out), generator=torchGenerator)
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
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)

  def __call__(self, x):
    if self.training:
      xmean = x.mean(0, keepdim=True)
      xvar = x.var(0, keepdim=True)
    else:
      xmean = self.running_mean
      xvar = self.running_var

    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
    self.out = self.gamma * xhat + self.beta
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
  

torchGenerator = torch.Generator().manual_seed(2147483647)
neuronCount = 200
featureDimensions = 10
matrix2ndDimensionSize = featureDimensions * blockSize
C = torch.randn((vocabSize, featureDimensions), generator=torchGenerator) #Embedding table
layers = [
  Linear(featureDimensions * blockSize, neuronCount, bias=False), BatchNorm1d(neuronCount), Tanh(),
  Linear(           neuronCount, neuronCount, bias=False), BatchNorm1d(neuronCount), Tanh(),
  Linear(           neuronCount, neuronCount, bias=False), BatchNorm1d(neuronCount), Tanh(),
  Linear(           neuronCount, neuronCount, bias=False), BatchNorm1d(neuronCount), Tanh(),
  Linear(           neuronCount, neuronCount, bias=False), BatchNorm1d(neuronCount), Tanh(),
  Linear(           neuronCount, vocabSize, bias=False), BatchNorm1d(vocabSize),
]

with torch.no_grad():
  layers[-1].gamma *= 0.1
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
for p in parameters:
  p.requires_grad = True

w1 = torch.randn((matrix2ndDimensionSize, neuronCount), generator=torchGenerator) * (5/3)/(matrix2ndDimensionSize)**0.5
w2 = torch.randn(neuronCount, vocabSize, generator=torchGenerator) * 0.01
b2 = torch.randn(vocabSize, generator=torchGenerator) * 0

bnGain = torch.ones((1, neuronCount))
bnBias = torch.zeros((1, neuronCount))
bnMeanRunning = torch.ones((1, neuronCount))
bnStdRunning = torch.zeros((1, neuronCount))



startTime = time.time()
batchSize = 32
maxSteps = 10000
for i in range(maxSteps):
  ix = torch.randint(0, xTrain.shape[0], (batchSize,))
  embeddings = C[xTrain[ix]]
  x = embeddings.view(embeddings.shape[0], -1)
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, yTrain[ix])
  
  for p in parameters:
    p.grad = None
  loss.backward()
  
  if(i > 20000):
    lr = 0.01
  else:
    lr = 0.1
  for p in parameters:
    p.data += -lr * p.grad

print('Time to train model in seconds: ', time.time() - startTime)

with torch.no_grad():
  embeddings = C[xTrain]
  hiddenLayerPreAct = (embeddings.view(embeddings.shape[0], -1)) @ w1
  batchNormMean = hiddenLayerPreAct.mean(0, keepdim=True)
  batchNormStd = hiddenLayerPreAct.std(0, keepdim=True)
  
def dataSplitLoss(split):
  x,y = {
    'train': (xTrain, yTrain),
    'val': (xDev, yDev)
  }[split]
  embeddings = C[x]
  x = embeddings.view(embeddings.shape[0], -1)
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, y)
  print(split, loss.item())

dataSplitLoss('train')
dataSplitLoss('val')


out = []
for i in range(1):
  singleName = []
  context = [0] * blockSize
  while True:
    embeddings = C[torch.tensor([context])]
    x = embeddings.view(embeddings.shape[0], -1)
    for layer in layers[:1]:
      x = layer(x)
      print(x)
    logits = x
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1, generator=torchGenerator).item()
    context = context[1:] + [ix]
    singleName.append(intToChar[ix])
    if ix == 0:
      break
  out.append(''.join(singleName))

print(out)
