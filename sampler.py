import random
import torch

def sample_patches(model_a, model_b, data, k):
    n = len(data)

    rows = random.sample(range(n), k)
    cols = random.sample(range(n), k)

    return return_patches(model_a, model_b, data, rows, cols)

def sample_square(model_a, model_b, data, k):
    n = len(data)

    idx = random.sample(range(n), k)
    return return_patches(model_a, model_b, data, idx, idx)


def return_patches(model_a, model_b, data, rows, cols):
    emb_a_rows = [model_a.embed(data[i]) for i in rows]
    emb_a_cols = [model_a.embed(data[j]) for j in cols]

    emb_b_rows = [model_b.embed(data[i]) for i in rows]
    emb_b_cols = [model_b.embed(data[j]) for j in cols]

    a_list = [torch.dot(x, y) for x in emb_a_rows for y in emb_a_cols]
    b_list = [torch.dot(x, y) for x in emb_b_rows for y in emb_b_cols]

    return a_list, b_list


