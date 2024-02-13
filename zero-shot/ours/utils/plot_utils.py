import matplotlib.pyplot as plt
import numpy as np

def plot_loss(train_loss, val_loss, epoch):
    plt.plot(train_loss, label='Train loss')
    plt.plot([loss.detach().cpu().numpy() for loss in val_loss], label='Val loss')
    plt.axhline(y=0.001, color='r', linestyle='-')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'{epoch}_loss.png')
    plt.close()

def plot_lr(lr, epoch):
    plt.plot(lr, label='Learning rate')
    plt.legend()
    plt.savefig(f'{epoch}_lr.png')
    plt.close()

def plot_series(data, pred, vdata, vpred, epoch, path):
    fig = plt.figure()
    fig.suptitle('Train and Val samples and its predictions')
    # gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    # (ax1, ax2), (ax3, ax4) = plt.subplots(sharex='col', sharey='row')
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig, ax = plt.subplots(2, 2)

    series_len = data['mask_series'][0].count_nonzero()
    series = data['series'][0].numpy()[:series_len]
    ax[0, 0].plot(series, label='series', color='blue')
    target_len = data['mask_target'][0].count_nonzero()
    targets = data['target'][0].numpy()[:target_len]
    ax[0, 0].plot([np.NaN] * len(series) + list(targets), label='targets', color='red')
    pred = pred.detach().cpu().numpy()[0][:target_len]
    ax[0, 0].plot([np.NaN] * len(series) + list(pred), label='pred', color='green')
    ax[0, 0].set_title('Train')
    # ax1.legend()

    ax[0, 1].plot([np.NaN] * len(series) + list(targets), color='red')
    ax[0, 1].plot([np.NaN] * len(series) + list(pred), color='green')
    # ax2.legend()
    ax[0, 1].axis([len(series) + 1, len(series) + len(targets) / 2, np.min(targets) - 1e-3, np.max(targets) + 1e-3])

    vseries_len = vdata['mask_series'][0].count_nonzero()
    vseries = vdata['series'][0].numpy()[:vseries_len]
    ax[1, 0].plot(vseries, color='blue')
    vtarget_len = vdata['mask_target'][0].count_nonzero()
    vtargets = vdata['target'][0].numpy()[:vtarget_len]
    ax[1, 0].plot([np.NaN] * len(vseries) + list(vtargets), color='red')
    vpred = vpred.detach().cpu().numpy()[0][:vtarget_len]
    ax[1, 0].plot([np.NaN] * len(vseries) + list(vpred), color='green')
    ax[1, 0].set_title('Validation')
    # ax3.legend()

    ax[1, 1].plot([np.NaN] * len(vseries) + list(vtargets), color='red')
    ax[1, 1].plot([np.NaN] * len(vseries) + list(vpred), color='green')
    # ax4.legend()
    ax[1, 1].axis([len(vseries) + 1, len(vseries) + len(vtargets) / 2, np.min(vtargets) - 1e-3, np.max(vtargets) + 1e-3])

    fig.tight_layout()
    # ax = ((ax1, ax2), (ax3, ax4))
    # handles, labels = ax.get_legend_handles_labels()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)
    # fig.legend(handles, labels, loc='upper center')
  
    plt.savefig(path + f'/series_{epoch}.png')
    plt.close()

def plot_series_(vdata, vpred, i, path, loss):
    fig = plt.figure()
    fig.suptitle('Train samples and its predictions')
    # gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    # (ax1, ax2), (ax3, ax4) = plt.subplots(sharex='col', sharey='row')
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig, ax = plt.subplots(2, 1)

    vseries_len = vdata['mask_series'][0].count_nonzero()
    vseries = vdata['series'][0].numpy()[:vseries_len]
    ax[0].plot(vseries, color='blue')
    vtarget_len = vdata['mask_target'][0].count_nonzero()
    vtargets = vdata['target'][0].numpy()[:vtarget_len]
    ax[0].plot([np.NaN] * len(vseries) + list(vtargets), color='red')
    vpred = vpred.detach().cpu().numpy()[0][:vtarget_len]
    ax[0].plot([np.NaN] * len(vseries) + list(vpred), color='green')
    ax[0].set_title(f'Train loss = {np.round(loss, 5)}.')
    # ax3.legend()

    ax[1].plot([np.NaN] * len(vseries) + list(vtargets), color='red')
    ax[1].plot([np.NaN] * len(vseries) + list(vpred), color='green')
    # ax4.legend()
    ax[1].axis([len(vseries) + 1, len([np.NaN] * len(vseries) + list(vtargets)) / 2, -2, 2])

    fig.tight_layout()
    # ax = ((ax1, ax2), (ax3, ax4))
    # handles, labels = ax.get_legend_handles_labels()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)
    # fig.legend(handles, labels, loc='upper center')
  
    plt.savefig(path + f'/series_{i}.png')
    plt.close()