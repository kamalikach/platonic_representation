import random
import torch

def sample_patches(model_a, model_b, data, k, device="cpu", seed=0,
                   transform_a=None, transform_b=None):
    random.seed(seed)
    n = len(data)

    rows = random.sample(range(n), k)
    cols = random.sample(range(n), k)

    return return_patches(model_a, model_b, data, rows, cols, device,
                          transform_a, transform_b)

def sample_square(model_a, model_b, data, k, device="cpu"):
    n = len(data)

    idx = random.sample(range(n), k)
    return return_patches(model_a, model_b, data, idx, idx)


def return_patches(model_a, model_b, data, rows, cols, device,
                   transform_a=None, transform_b=None):
    if transform_a is None:
        transform_a = lambda x: x
    if transform_b is None:
        transform_b = lambda x: x

    emb_a_rows = [model_a.embed(transform_a(data[i]).to(device)) for i in rows]
    emb_a_cols = [model_a.embed(transform_a(data[j]).to(device)) for j in cols]

    emb_b_rows = [model_b.embed(transform_b(data[i]).to(device)) for i in rows]
    emb_b_cols = [model_b.embed(transform_b(data[j]).to(device)) for j in cols]

    a_list = [torch.dot(x, y).item() for x in emb_a_rows for y in emb_a_cols]
    b_list = [torch.dot(x, y).item() for x in emb_b_rows for y in emb_b_cols]

    return a_list, b_list
