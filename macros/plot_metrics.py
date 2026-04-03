import json
import numpy as np
import matplotlib.pyplot as plt
import re

input_dir = '/nfs/data/1/nitish/dino_output/cvn_minkunet_trywarp2/'
#  input_dir = '/nfs/data/1/nitish/dino_output/cvn_properrun_1gpu/'
out_dir = input_dir+'plots/'

f_train_metrics = input_dir + 'training_metrics.json'
f_grads = input_dir + 'grads/grad_history_10epoch0001.json'

with open(f_train_metrics, 'r') as f:
    metrics = None
    for i, line in enumerate(f):
        metrics_dict = json.loads(line)
        dtype = [(key, type(metrics_dict[key])) for key in metrics_dict.keys()]
        # print(dtype)
        if i==0:
            metrics = np.array([tuple(metrics_dict.values())], dtype=dtype)
        else:
            metrics = np.append(metrics, np.array([tuple(metrics_dict.values())], dtype=dtype), axis=0)
aux = ['masked_patches_ratio']
losses_keys = [k for k in metrics.dtype.names if 'loss' in k or k in aux]
print(losses_keys)

fig, axes = plt.subplots(3, 2, figsize=(5*2, 4*3))

for i, k in enumerate(losses_keys):
    if i < 6:  # Only plot first 6 losses (3x2 = 6 subplots)
        row = i // 2  # Calculate row index
        col = i % 2   # Calculate column index
        print(metrics[k].shape)
        axes[row, col].plot(metrics['iteration'], metrics[k], label=k)
        axes[row, col].set_xlabel('Iteration', fontsize=15)
        axes[row, col].set_ylabel('loss', fontsize=15)
        axes[row, col].grid(True)
        axes[row, col].legend(fontsize=15)

# Hide empty subplots if you have fewer than 6 losses
for i in range(len(losses_keys), 6):
    row = i // 2
    col = i % 2
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(out_dir+'metrics.pdf')

def plot_grad(epoch=9):
    f_grads = input_dir + 'grads/grad_history_10epoch000%d.json' % epoch
    grad_metrics = {}
    ignore = ['bias']
    with open(f_grads, 'r') as f:
        try:
            grads = json.load(f)
        except json.decoder.JSONDecodeError:
            return
        for key in grads.keys():
            if np.any([ig in key for ig in ignore]): continue
            grad_metrics[key] = np.array(grads[key]['norms'])


    print(list(grad_metrics.keys()))
    print(len(grad_metrics.keys()))
    fig, axes = plt.subplots(7, 4, figsize=(7*4, 4*7))
    for i, key in enumerate(grad_metrics.keys()):
        row = i // 4
        col = i % 4
        #  if row >= 5: continue
        indices = np.arange(len(grad_metrics[key]))
        axes[row, col].plot(indices[:80], grad_metrics[key][:80], label=re.sub(r'_fsdp_wrapped_module', '', key))
        axes[row, col].set_xlabel('Iteration', fontsize=15)
        axes[row, col].set_ylabel('Gradient (L2-Norm)', fontsize=15)
        axes[row, col].grid(True)
        axes[row, col].legend(fontsize=15)

    plt.tight_layout()
    plt.savefig(out_dir+'grads_epoch%d_zoom.pdf'%epoch)

#  plot_grad(1)
#  for i in range(9):
#      plot_grad(i)
