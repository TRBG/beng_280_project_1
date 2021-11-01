import argparse
import matplotlib. pyplot as plt
import numpy as np

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LogMeter(object):
    """Logging class used to count and stores aggregates and means"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_scans_and_reconstructions(output, target):
    
    output1 = output.detach().cpu().numpy()
    target1 = target.detach().cpu().numpy()
    
   # Plotting the ground truth scans
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,10))

    trgt_gt_1 = ((target1[0,:,:,:]-np.min(target1[0,:,:,:]))/(np.max(target1[0,:,:,:]-np.min(target1[0,:,:,:])))).copy()
    im1 = ax1.imshow(trgt_gt_1.transpose(1, 2, 0)[:,:,0])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    trgt_gt_2 = ((target1[1,:,:,:]-np.min(target1[1,:,:,:]))/(np.max(target1[1,:,:,:]-np.min(target1[1,:,:,:])))).copy()
    im2 = ax2.imshow(trgt_gt_2.transpose(1, 2, 0)[:,:,0])
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    trgt_gt_3 = ((target1[2,:,:,:]-np.min(target1[2,:,:,:]))/(np.max(target1[2,:,:,:]-np.min(target1[2,:,:,:])))).copy()
    im3 = ax3.imshow(trgt_gt_3.transpose(1, 2, 0)[:,:,0])
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
    
    # Plotting the reconstructed scans
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,10))

    recon_1 = ((output1[0,:,:,:]-np.min(output1[0,:,:,:]))/(np.max(output1[0,:,:,:]-np.min(output1[0,:,:,:])))).copy()
    im1 = ax1.imshow(recon_1.transpose(1, 2, 0)[:,:,0])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    recon2 = ((output1[1,:,:,:]-np.min(output1[1,:,:,:]))/(np.max(output1[1,:,:,:]-np.min(output1[1,:,:,:])))).copy()
    im2 = ax2.imshow(recon2.transpose(1, 2, 0)[:,:,0])
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    recon3 = ((output1[2,:,:,:]-np.min(output1[2,:,:,:]))/(np.max(output1[2,:,:,:]-np.min(output1[2,:,:,:])))).copy()
    im3 = ax3.imshow(recon3.transpose(1, 2, 0)[:,:,0])
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

###############################################################################################################
