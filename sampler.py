import random
import torch

def sample_patches(model_a, model_b, data, k, device="cpu"):
    n = len(data)

    rows = random.sample(range(n), k)
    cols = random.sample(range(n), k)

    return return_patches(model_a, model_b, data, rows, cols, device)

def sample_square(model_a, model_b, data, k, device="cpu"):
    n = len(data)

    idx = random.sample(range(n), k)
    return return_patches(model_a, model_b, data, idx, idx)


def return_patches(model_a, model_b, data, rows, cols, device):
    emb_a_rows = [model_a.embed(data[i].to(device)) for i in rows]
    emb_a_cols = [model_a.embed(data[j].to(device)) for j in cols]

    emb_b_rows = [model_b.embed(data[i].to(device)) for i in rows]
    emb_b_cols = [model_b.embed(data[j].to(device)) for j in cols]

    a_list = [torch.dot(x, y).item() for x in emb_a_rows for y in emb_a_cols]
    b_list = [torch.dot(x, y).item() for x in emb_b_rows for y in emb_b_cols]

    return a_list, b_list


