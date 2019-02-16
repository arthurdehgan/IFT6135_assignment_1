import numpy as np
import torch
import pandas as pd
from network import Net

if __name__ == "__main__":
    submission_data = np.load("test_data.npy")
    submission_data = submission_data / 255.0

    infos = np.load("best_model_info.npy").reshape(1)[0]
    net = Net(3, infos["convs"], infos["lins"])
    net.load_state_dict(torch.load("best_model_net"))

    y = []
    split = 10
    step = int(len(submission_data) / split)
    for i in range(0, len(submission_data), step):
        batch = torch.Tensor(submission_data[i : i + step])
        batch = batch.view(-1, 3, 64, 64)
        batch = batch.cuda()

        y += list(map(int, net.forward(batch).max(1)[1].cpu()))

    y = pd.DataFrame(y)
    y.index.name = "id"
    y.columns = ["label"]
    y = y.replace(0, "Cat")
    y = y.replace(1, "Dog")
    y.index += 1
    y.to_csv("submission.csv")
