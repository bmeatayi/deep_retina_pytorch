import torch
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


class PlotProps:
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

    def __init__(self):
        pass

    # Set figure and subplot properties

    def init_figure(self, figsize, hspace=.3, wspace=.1):
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        return fig

    def init_subplot(self, title,
                     tot_tup=(1, 1), sp_tup=(0, 0),
                     colspan=1, rowspan=1,
                     sharex=None, sharey=None,
                     xlabel='', ylabel='',
                     despine=True,
                     offset=5, trim=False, 
                     ttl_fs=15, ttl_pos='center'):

        ax = plt.subplot2grid(tot_tup, sp_tup, colspan, rowspan, sharex=sharex, sharey=sharey)

        ax.set_title(title, fontsize=ttl_fs, loc=ttl_pos)

        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel(ylabel, fontsize=15)

        sns.set(context='paper', style='ticks', font_scale=1.5)
        sns.axes_style({'axes.edgecolor': '.6', 'axes.linewidth': 5.0})
        if despine is True:
            sns.despine(ax=ax, offset=offset, trim=trim)

        return ax

    def legend(self, loc='best', fontsize=15):
        plt.legend(loc=loc, fontsize=fontsize, frameon=False)
        

def evaluate_model(model, dataset, fr_start=0, fr_end=1000):
    stim = np.zeros((0, model.nL, model.nW, model.nH))
    gt = np.zeros((0, model.nCell))
    pred = np.zeros((0, model.nCell))
    model.eval()
    model.cuda()
    for idx in range(fr_start, fr_end):
        stim, gt_fr = dataset[idx]
        stim = torch.from_numpy(stim[np.newaxis, :, :, :]).float().cuda()
        pred_temp = model(stim)
        pred_temp = pred_temp.data.cpu().numpy()
        pred = np.concatenate((pred, pred_temp), axis=0)
        gt = np.concatenate((gt, gt_fr), axis=0)
    model.train()
    return gt, pred


def plot_results(model, cell_idx, gt_tr, pred_tr, gt_val, pred_val, val_start, tr_start, sta_rf=None, outfile='results'):
    t = range(tr_start, tr_start+pred_tr.shape[0])
    pdf = PdfPages(outfile + '//results_cell_'+str(cell_idx)+'.pdf')
    myPlot = PlotProps()
    fig = myPlot.init_figure(figsize=(20, 10), hspace=.3, wspace=.1)
    ax1 = myPlot.init_subplot(title='Cell No.'+str(cell_idx)+' (Train dataset)',
                              tot_tup=(2, 1), sp_tup=(0, 0),
                              colspan=1, rowspan=1,
                              sharex=None, sharey=None,
                              xlabel='Frames', ylabel='Spike counts',
                              despine=True,
                              offset=5, trim=False,
                              ttl_fs=15, ttl_pos='center')
    ax1.plot(t, gt_tr[:, cell_idx], 'r', label='Groundtruth')
    ax1.plot(t, pred_tr[:, cell_idx], 'b', label='CNN prediction')
    myPlot.legend()
    # plt.ylim((0, 120))
    # plt.show()
    # pdf.savefig(fig)
    
    t = range(val_start, val_start+pred_val.shape[0])
    # myPlot = PlotProps()
    # fig = myPlot.init_figure(figsize=(20, 5), hspace=.3, wspace=.1)
    ax2 = myPlot.init_subplot(title='Cell No. '+str(cell_idx)+ ' (Validation dataset)',
                              tot_tup=(2, 1), sp_tup=(1, 0),
                              colspan=1, rowspan=1,
                              sharex=None, sharey=None,
                              xlabel='Frame', ylabel='Spike counts',
                              despine=True,
                              offset=5, trim=False,
                              ttl_fs=15, ttl_pos='center')
    ax2.plot(t, gt_val[:, cell_idx], 'r', label='Groundtruth')
    ax2.plot(t, pred_val[:, cell_idx], 'b', label='CNN prediction')
    myPlot.legend()
    # plt.ylim((0, 120))
    pdf.savefig(fig)

    # Plot weights of the fully connected layer
    layer3_weights = np.reshape(list(model.fc.parameters())[0].cpu().detach().numpy(),
                                model.l3_filt_shape)[cell_idx, :, :, :]
    
    fig, axes = plt.subplots(1, layer3_weights.shape[0], figsize=(20, 3), sharex=True, sharey=True)
    fig.suptitle('Visualization of weights in fully connected layer', fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    vmin = layer3_weights.min()
    vmax = layer3_weights.max()

    for filt_t, ax in zip(layer3_weights, axes):
        ax.imshow(filt_t, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
    pdf.savefig(fig)

    if sta_rf is not None:
        fig, axes = plt.subplots(1, layer3_weights.shape[0], figsize=(20, 3), sharex=True, sharey=True)
        fig.suptitle('Receptive fields obtained by STA', fontsize=20)
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        vmin = sta_rf.min()
        vmax = sta_rf.max()

        for filt_t, ax in zip(sta_rf, axes):
            ax.imshow(filt_t, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
        pdf.savefig(fig)
    
    plt.show()
    pdf.close()


def visualize_deep_retina(model, outfile='results'):
    layer1_filters = list(model.conv1.parameters())[0].cpu().detach().numpy()
    layer1_bias = list(model.conv1.parameters())[1].cpu().detach().numpy()

    layer2_filters =  list(model.conv2.parameters())[0].cpu().detach().numpy()
    layer2_bias = list(model.conv2.parameters())[1].cpu().detach().numpy()

    layer3_weights = np.reshape(list(model.fc.parameters())[0].cpu().detach().numpy(), model.l3_filt_shape)
    layer3_bias = list(model.fc.parameters())[1].cpu().detach().numpy()
    softPlus_param = list(model.act_function.parameters())[0].cpu().detach().numpy()
    n1_filt = layer1_filters.shape[0]
    n1_length = layer1_filters.shape[1]
    
    n2_filt = layer2_filters.shape[0]
    n2_length = layer2_filters.shape[1]

    n3_filt = layer3_weights.shape[0]
    n3_length = layer3_weights.shape[1]

    pdf = PdfPages(outfile+ 'filter_vis.pdf')

    # Plot filters of convolutional layer 1
    vmin = layer1_filters.min()
    vmax = layer1_filters.max()
    fig, axes = plt.subplots(n1_length, n1_filt, figsize=(20, 8), sharex=True, sharey=True)
    fig.suptitle('Visualization of filters in first layer', fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.97)
    for filt_i, ax_row in zip(np.transpose(layer1_filters, (1, 0, 2, 3)), axes):
        # vmin = filt_i.min()
        # vmax = filt_i.max()
        for filt_t, ax in zip(filt_i, ax_row):
            mapable = ax.imshow(filt_t, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(mapable, ax=ax, orientation='vertical', fraction=.1)
    pdf.savefig(fig)

    # Plot filters of convolutional layer 2
    vmin = layer2_filters.min()
    vmax = layer2_filters.max()
    fig, axes = plt.subplots(n2_filt, n2_length, figsize=(20, 15), sharex=True, sharey=True)
    fig.suptitle('Visualization of filters in second layer', fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.96)
    for filt_i, ax_row in zip(np.transpose(layer2_filters, (0, 1, 2, 3)), axes):
        # vmin = filt_i.min()
        # vmax = filt_i.max()
        for filt_t, ax in zip(filt_i, ax_row):
            mapable = ax.imshow(filt_t, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(mapable, ax=ax, orientation='vertical', fraction=.1)
    pdf.savefig(fig)

    # Plot weights of the fully connected layer
    vmin = layer3_weights.min()
    vmax = layer3_weights.max()
    fig, axes = plt.subplots(n3_filt, n3_length, figsize=(20, n3_filt*2), sharex=True, sharey=True)
    fig.suptitle('Visualization of weights in third layer', fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.97)
    for filt_i, ax_row in zip(layer3_weights, axes):
        # vmin = filt_i.min()
        # vmax = filt_i.max()
        for filt_t, ax in zip(filt_i, ax_row):
            mapable = ax.imshow(filt_t, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(mapable, ax=ax, orientation='vertical', fraction=.1)
    pdf.savefig(fig)
    pdf.close()