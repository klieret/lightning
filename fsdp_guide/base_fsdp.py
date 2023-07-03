import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.demos import Transformer, WikiText2
from lightning.fabric.strategies import FSDPStrategy

from functools import partial
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


policy = partial(
    transformer_auto_wrap_policy, 
    transformer_layer_cls={nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}
)

strategy = FSDPStrategy(
    auto_wrap_policy=policy,
    activation_checkpointing=[
        nn.TransformerEncoderLayer, 
        nn.TransformerDecoderLayer,
    ],
    cpu_offload=True,
)

fabric = L.Fabric(devices=2, strategy=strategy)
fabric.launch()

with fabric.rank_zero_first():
    dataset = WikiText2()

with fabric.init_module(empty_init=False):
    model = Transformer(vocab_size=dataset.vocab_size, nlayers=32, nhid=4096, ninp=1024, nhead=64)

optimizer = torch.optim.Adam(model.parameters())

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
