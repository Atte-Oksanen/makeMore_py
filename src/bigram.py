# A simple bigram model for looking up next characters in a sequence
import torch
namelist = open('C:/Users/Atte/Documents/Ylijopisto/makeMore_py/src/names.txt', 'r').read().splitlines()
ab = torch.zeros((27,27), dtype=torch.int32)
chars = sorted(list(set(''.join(namelist))))
charToInt = {s: i + 1 for i,s in enumerate(chars)}
charToInt['.'] = 0
intToChar = {i: s for s,i in charToInt.items()}
for w in namelist:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = charToInt[ch1]
    ix2 = charToInt[ch2]
    ab[ix1, ix2] += 1
P = (ab + 1).float()
P /= P.sum(1 , keepdim=True)

ix = 0
out = []
torchGenerator = torch.Generator().manual_seed(2147483647)
for i in range(10):
  name = []
  while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=torchGenerator).item()
    name.append(intToChar[ix])
    if ix == 0:
      break
  out.append(''.join(name))
print('\n'.join(out))


logLikelyhood = 0.0
n = 0
for w in namelist:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = charToInt[ch1]
    ix2 = charToInt[ch2]
    prob = P[ix1, ix2]
    logProb = torch.log(prob)
    logLikelyhood += logProb
    n += 1
nll = -logLikelyhood
nllAvg = nll / n
print(nll)
print(nllAvg)