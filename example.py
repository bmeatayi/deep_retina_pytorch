import torch
from util.dataset import SpikeDataset
from model.deep_retina import DeepRetina
from util.solver import Solver
import torch.utils.data.dataloader as dataloader


cnt_file = 'dataset//primer//cnt//'
stim_file = 'dataset//primer//'
# Whole dataset length: 137145
tr_start_frame = 0
tr_end_frame = 82000

val_start_frame = 82000
val_end_frame = 110000

cell_idx_list = list(range(0, 53))
nCells = len(cell_idx_list)

train_dataset = SpikeDataset(spike_rate_path=cnt_file,
                             stim_path=stim_file,
                             cell_idx=cell_idx_list,
                             win_len=5,
                             n_split=1,
                             fr_st_end=(tr_start_frame, tr_end_frame))

val_dataset = SpikeDataset(spike_rate_path=cnt_file,
                           stim_path=stim_file,
                           cell_idx=cell_idx_list,
                           n_split=1,
                           win_len=5,
                           fr_st_end=(val_start_frame, val_end_frame))

model = DeepRetina(nw=40, nh=40, nl=5,
                   n_filters=(8, 8),
                   kernel_size=(9, 7),
                   n_cell=nCells)

train_dataloader = dataloader.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
val_dataloader = dataloader.DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True)


solver = Solver(optim_args={"lr": 1e-4},
                loss_func=torch.nn.PoissonNLLLoss(log_input=False, full=True),
                l1_w=1e-6, l2_w=1e-6, log_folder='logs//run1')

solver.train(model,
             train_loader=train_dataloader,
             val_loader=val_dataloader,
             log_nth=200,
             num_epochs=50)

