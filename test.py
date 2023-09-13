# Author: Christos Gklezos

import ADWSN.cluster_run as mynn
import torch

# model = mynn.loadModel()
#
# print(torch.sum(mynn.infer(torch.tensor([[20, 21., 20, 20, 20, 40],
#                                          [10, 11, 10, 10, 10, 45]]), model)))
# print(model.extract_enc())



anomaly = torch.tensor([[20, 21., 24, 20, 20, 40],
                        [20, 21, 20, 20, 20, 10],
                        [20, 21, 20, 20, 17, 40],
                        [10, 21, 20, 20, 20, 40],
                        [20, 21, 23, 20, 20, 40],
                        [20, 21, 18, 20, 20, 40],
                        [20, 21, 20, 20, 22, 40],
                        [18, 21, 20, 20, 22, 40]], dtype=torch.float32)

normal = torch.tensor([[20, 21, 20, 20, 20, 40],
                       [15, 16, 15, 15, 15, 42.5],
                       [17, 18, 17, 17, 17, 42],
                       [11, 12, 11, 11, 11, 45]], dtype=torch.float32)

# torch.save(normal, "normal_sample.pt")
torch.save(anomaly, "anomaly_sample.pt")
