# A single layer neural network based on bigrams with gradient based optimization
import torch
import torch.nn.functional as F

namelist = open('C:/Users/Atte/Documents/Ylijopisto/makeMore_py/src/names.txt', 'r').read().splitlines()
ab = torch.zeros((27,27), dtype=torch.int32)
chars = sorted(list(set(''.join(namelist))))

charToInt = {s: i + 1 for i,s in enumerate(chars)}
charToInt['.'] = 0
intToChar = {i: s for s,i in charToInt.items()}

xs, ys = [], [] 
for w in namelist:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    xs.append(charToInt[ch1])
    ys.append(charToInt[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = len(xs)
torchGenerator = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=torchGenerator, requires_grad=True)

for cycle in range(800):
  xEnc = F.one_hot(xs, num_classes=27).float()
  logits = xEnc @ W
  counts = logits.exp()
  probs = counts / counts.sum(1, keepdim=True)
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
  W.grad = None
  loss.backward()
  W.data += -25 * W.grad
  if(cycle % 20 == 0):
    print('network loss: ', loss.item())
    print('cycles: ', cycle)


  out = []
for i in range(5):
  singleName = []
  ix = 0
  while True:
    xEncoded = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xEncoded @ W
    counts = logits.exp()
    p = counts / counts.sum(1, keepdims=True)
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=torchGenerator).item()
    singleName.append(intToChar[ix])
    if ix == 0:
      break
  out.append(''.join(singleName))
print('\n'.join(out))