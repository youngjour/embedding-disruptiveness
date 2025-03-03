# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 14:33:29
# @Last Modified by:   Munjung Kim
# @Last Modified time: 2023-06-03 22:41:24
import torch
from tqdm import tqdm
from torch.optim import AdamW, Adam, SGD, SparseAdam
from torch.utils.data import DataLoader


def train(
    model,
    dataset,
    loss_func,
    batch_size=256 * 4,
    checkpoint=10000,
    outputfile=None,
    learning_rate=1e-3,
    num_workers=1,
):

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Training
    
    focal_params = filter(lambda p: p.requires_grad, model.parameters())
    optim = SparseAdam(focal_params, lr=learning_rate)

    pbar = tqdm(dataloader, miniters=10, total=len(dataloader))
    it = 0
    for params in pbar:
        # clear out the gradient
        optim.zero_grad()
        

        # compute the loss
        loss = loss_func(model, *params)

        # backpropagate
        loss.backward()

        # update the parameters
        optim.step()

        with torch.no_grad():
            pbar.set_postfix(loss=loss.item())

            if (it + 1) % checkpoint == 0:
                if outputfile is not None:
                    torch.save(model.state_dict(), outputfile)
        it += 1
        torch.cuda.empty_cache()
    if outputfile is not None:
        torch.save(model.state_dict(), outputfile)
    model.eval()
    return model
