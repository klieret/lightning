import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.demos import Transformer, WikiText2

fabric = L.Fabric(devices=2, strategy="ddp")
fabric.launch()

with fabric.rank_zero_first():
    dataset = WikiText2()

model = Transformer(vocab_size=dataset.vocab_size, nlayers=32, nhid=4096, ninp=1024, nhead=64)
optimizer = torch.optim.Adam(model.parameters())

print(f"{sum(p.numel() for p in model.parameters()) / 1e9:.3f} B")

model, optimizer = fabric.setup(model, optimizer)

for i in range(10):
    input, target = fabric.to_device(dataset[i])
    output = model(input.unsqueeze(0), target.unsqueeze(0))
    loss = F.nll_loss(output, target.view(-1))
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    fabric.print(loss.item())

fabric.print(torch.cuda.memory_summary())
