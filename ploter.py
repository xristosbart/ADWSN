# Filename: ploter.py
# Author: Christos Gklezos


import torch, torchvision
from matplotlib import pyplot as plt
import ADWSN.cluster_run as myCluster
import ADWSN.cloud_run as myCloud
import ADWSN.cloud_run_diff as myCloudDiff

def plot_cluster():
    plt.figure(1)
    anomaly = torch.load("lab_anomalies_sample.pt")
    normal = torch.load("lab_sample_data_2.pt")[:10000]#"normal_sample.pt")
    model = myCluster.loadModel()
    normal_o = torch.sum(myCluster.infer(normal.float(), model), 1)
    anomaly_o = torch.sum(myCluster.infer(anomaly.float(), model), 1)
    print("Normal:", normal_o.shape[0], "\nAnomalies:", anomaly_o.shape[0])
    count = torch.sum(torch.gt(normal_o, 2)).item()
    print("Count: " + str(count))
    print("Anomaly ratio: {:2.2%}".format(count/normal_o.shape[0]))
    n_range = torch.arange(0, normal_o.shape[0], 1).unsqueeze(1)
    a_range = torch.arange(0, normal_o.shape[0], (normal_o.shape[0]) / anomaly_o.shape[0]).unsqueeze(1)
    plt.xlabel("Sample")
    plt.ylabel("Reconstruction Error")
    plt.yscale('log')
    plt.grid(True)  # which='both'
    # plt.minorticks_on()

    plt.plot(n_range.numpy(), normal_o.detach().numpy(), '.')
    plt.plot(a_range.numpy(), anomaly_o.detach().numpy(), 'r.')

def plot_cloud():
    plt.figure(2)
    anomaly = torch.load("cloud_anomalies.pt")
    normal_c = torch.load("cloud_train_data.pt")
    normal_c = torch.reshape(normal_c, (4000, 5, 72))
    normal_c = torch.reshape(normal_c, (20000, 72))
    cluster = myCluster.loadModel()
    model = myCloud.loadModel()

    myCluster.infer(anomaly, cluster)
    anomaly_c = cluster.extract_enc().flatten(1)
    normal_o = torch.sum(myCloud.infer(normal_c, model), 1)
    anomaly_o = torch.sum(myCloud.infer(anomaly_c, model), 1)
    n_range = torch.arange(0, 20000, 20000. / (normal_o.shape[0])).unsqueeze(1)
    a_range = torch.arange(0, 20000, 20000. / (anomaly_o.shape[0])).unsqueeze(1)
    plt.xlabel("Samples")
    plt.ylabel("Reconstruction Error")
    plt.yscale('log')
    plt.grid(True)  # which='both'
    # plt.minorticks_on()

    plt.plot(n_range.numpy(), normal_o.detach().numpy(), '.')
    plt.plot(a_range.numpy(), anomaly_o.detach().numpy(), 'r.')

def plot_cloud_diff():
    plt.figure(3)
    anomaly = torch.load("cloud_anomalies.pt")
    normal_c = torch.load("cloud_train_data.pt")
    normal_c = torch.reshape(normal_c, (4000, 5, 24, 3))
    normal_c = normal_c[:, :, 1:, :] - normal_c[:, :, :-1, :]
    normal_c = torch.reshape(normal_c, (460000, 3))

    cluster = myCluster.loadModel()
    model = myCloudDiff.loadModel()

    myCluster.infer(anomaly, cluster)
    anomaly_c = cluster.extract_enc()
    anomaly_c = anomaly_c[:, 1:, :] - anomaly_c[:, :-1, :]
    anomaly_c = anomaly_c.flatten(start_dim=0, end_dim=1)
    normal_o = torch.sum(myCloudDiff.infer(normal_c, model), 1)
    anomaly_o = torch.sum(myCloudDiff.infer(anomaly_c, model), 1)
    anomaly_o = anomaly_o[anomaly_o > 1e-2]
    n_range = torch.arange(0, normal_o.shape[0], 1).unsqueeze(1)
    a_range = torch.arange(0, normal_o.shape[0], normal_o.shape[0] / (anomaly_o.shape[0])).unsqueeze(1)
    plt.xlabel("Samples")
    plt.ylabel("Reconstruction Error")
    plt.yscale('linear')
    plt.grid(True)  # which='both'
    # plt.minorticks_on()

    plt.plot(n_range.numpy(), normal_o.detach().numpy(), '.')
    plt.plot(a_range.numpy(), anomaly_o.detach().numpy(), 'r.')

def plot_data(data_path):
    plt.figure(4)
    data = torch.load(data_path)#[1500:2500]
    # data = data[:, 1:, :] - data[:, :-1, :]
    # d_range = torch.arange(1500+1, 1500+data.shape[0]+1).unsqueeze(1) # P1
    d_range = torch.arange(1, data.shape[1] + 1).unsqueeze(1)  # P2
    #print(data.shape)
    # plt.plot(d_range, data[:, 0].detach().numpy(), '.') # P1
    plt.xlabel("Sample")
    plt.ylabel("Data Channel 0")
    plt.yscale('linear')
    plt.grid(True)  # which='both'
    # plt.minorticks_on()
    plt.plot(d_range, data[7, :, 0].detach().numpy(), '.')  # P2


plot_data('cloud_anomalies.pt')# "lab_sample_data_2.pt")
# plot_cluster()
plot_cloud()
plot_cloud_diff()
plt.show()