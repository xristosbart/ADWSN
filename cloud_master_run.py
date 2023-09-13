import torch


def join_clusters_1D(clusters):
    #first should be 1 value time, followed by each cluster's channels 1D, n sized tensors
        return torch.cat(clusters, dim=1)


def create_window(frames):
    # 10 N sized frames created by join_clusters_1D
    return torch.flatten(frames)


def update_window(window, new_frame):
    return torch.cat((window[new_frame.size()[0]:], new_frame))


def main():
    # count = 0
    # while True
        # load data
        # join_clusters
        # run diff and simple model
        # if count % 24 == 0:
            # create_window
            # run cloud (window)
    pass


if __name__ == "__main__":
    main()

