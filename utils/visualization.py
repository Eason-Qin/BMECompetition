import matplotlib.pyplot as plt
import torch
import numpy as np
def print_cut_samples(data,label,predict,save_path):
    z = np.any(torch.argmax(predict, dim=1).detach().cpu(), axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]
    cut=abs(start_slice-end_slice)/2
    plt.figure("check", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("image")
    plt.imshow(data.cpu().numpy()[0, 0, :, :, cut], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("label")
    plt.imshow(label.cpu().numpy()[0, 0, :, :, cut])
    plt.subplot(1, 3, 3)
    plt.title("output")
    plt.imshow(torch.argmax(predict, dim=1).detach().cpu()[0, :, :, cut])
    plt.savefig(save_path,dpi=300)



def print_heatmap(data,label,predict,grad_graph,save_path):
    z = np.any(torch.argmax(predict, dim=1).detach().cpu(), axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]
    cut=abs(start_slice-end_slice)/2
    plt.figure("check", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("image")
    plt.imshow(data.cpu().numpy()[0, 0, :, :, cut], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("label")
    plt.imshow(label.cpu().numpy()[0, 0, :, :, cut])
    plt.subplot(1, 3, 3)
    plt.title("heat")
    plt.imshow(grad_graph[0, :, :, cut])
    plt.savefig(save_path,dpi=300)
    



