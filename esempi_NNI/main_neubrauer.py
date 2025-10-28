"""
Script for training (with validation) and test of a RSNN, written in snnTorch
for Braille (100 Hz or 40 Hz, to be MANUALLY SET) classification,
for optimization through NNI.

Settings to be accounted for:
    experiment_name
    experiment_data
    Braille_path
    encoding_populations_needed
    output_pop
    nb_hidden
    alpha_hid
    alpha_out
    beta_hid
    beta_out
    beta_enc
    slope
    n_test
    nb_epochs
    lr
    reg_l1
    reg_l2
    save_weights
    store_weights
    use_seed
    batch_size

Fra, Vittorio,
Politecnico di Torino,
EDA Group,
Torino, Italy.
"""

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch.functional import quant

import nni
from nni.tools.nnictl import updater

import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
parentDir = os.path.dirname(fileDir)
if parentDir not in sys.path:
    sys.path.append(parentDir)
from utils.device_manager import *
from utils.data_manager import *
from utils.quant_onnx import * # customized state_quant function
from data.load_Braille import *
sys.path.remove(parentDir)
sys.path.remove(os.path.join(parentDir,"NNI"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

import datetime
import logging
import argparse
import copy


experiment_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


###### 0) Various experiment default settings ##################################

### SET THE EXPERIMENT NAME and check default settings in the parser to be possibly modified
experiment_name = "braille_letters_digits_40Hz_augmented_populations" # name of the experiment as in the "main" script for NNI configuration

searchspace_filename = "train_snnTorch_Braille_40Hz_searchspace"
searchspace_path = "./searchspaces/{}.json".format(searchspace_filename)

class SearchSpaceUpdater(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

parser = argparse.ArgumentParser()

# Experiment data
parser.add_argument('-experiment_data',
                    type=str,
                    default="Braille_40Hz_augmented",
                    help='Name describing what data are used for the experiment.')
# Filepath for Braille data
parser.add_argument('-Braille_path',
                    type=str,
                    default="../data/data_braille_letters_digits_40Hz_augmented",
                    help='Path for the Braille data to be loaded.')
# Encode through populations or not
parser.add_argument('-encoding_populations_needed',
                    type=bool,
                    default=True,
                    help='Set if encoding is to be performed through populations.')
# Populations dimension
parser.add_argument('-neurons_per_pop',
                    type=int,
                    default=10,
                    help='Number of neurons in each population of the encoding layer or total number of neurons in the encoding layer if populations are not needed.')
# Output "channels" of each population
parser.add_argument('-output_pop',
                    type=int,
                    default=1,
                    help='Output dimension of each population in the encoding layer.')
# Neurons in the hidden layer
parser.add_argument('-nb_hidden',
                    type=int,
                    default=450,
                    help='Size of the hidden layer.')
# Decay factor for synaptic current in the hidden layer
parser.add_argument('-alpha_hid',
                    type=float,
                    default=0.6,
                    help='Decay factor for synaptic current in the hidden layer.')
# Decay factor for synaptic current in the output layer
parser.add_argument('-alpha_out',
                    type=float,
                    default=0.4,
                    help='Decay factor for synaptic current in the output layer.')
# Decay factor for membrane potential in the hidden layer
parser.add_argument('-beta_hid',
                    type=float,
                    default=0.9,
                    help='Decay factor for membrane potential in the hidden layer.')
# Decay factor for membrane potential in the output layer
parser.add_argument('-beta_out',
                    type=float,
                    default=0.8,
                    help='Decay factor for membrane potential in the output layer.')
# Decay factor for membrane potential in the encoding layer
parser.add_argument('-beta_enc',
                    type=float,
                    default=0.95,
                    help='Decay factor for membrane potential in the encoding layer.')
# Slope for the surrogate gradient
parser.add_argument('-slope',
                    type=int,
                    default=10,
                    help='Slope for the surrogate gradient.')
# Number or tests for statistics
parser.add_argument('-n_test',
                    type=int,
                    default=50,
                    help='Number of tests to be performed for statistical evaluation.')                                                       
# Number of epochs
parser.add_argument('-nb_epochs',
                    type=int,
                    default=500,
                    help='Number of training epochs.')
# Learning rate
parser.add_argument('-lr',
                    type=float,
                    default=1e-3,
                    help='Learning rate.')
# L1 loss on the total number of spikes
parser.add_argument('-reg_l1',
                    type=float,
                    default=1e-4,
                    help='Coefficient for L1-like regularizer on the total number of spikes.')
# L2 loss on the number of spikes per neuron
parser.add_argument('-reg_l2',
                    type=float,
                    default=1e-8,
                    help='Coefficient for L1-like regularizer on the number of spikes per neuron.')
# Save the weights from optimal results
parser.add_argument('-save_weights',
                    type=bool,
                    default=True,
                    help='Weights can be saved to be loaded after training and used for test.')
# Store the weights 
parser.add_argument('-store_weights',
                    type=bool,
                    default=True,
                    help='Weights can be stored with specific, unique name.')
# Set seed usage
parser.add_argument('-use_seed',
                    type=bool,
                    default=False,
                    help='Set if a seed is to be used or not.')
# Choose the batch size
parser.add_argument('-batch_size',
                    type=int,
                    default=128,
                    help='Specify the batch size.')               

args = parser.parse_args()

settings = vars(args)

use_seed = settings["use_seed"]

device = torch.device("cuda:0")

if use_seed:
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

else:
    seed = None


################################################################################


###### 1) Training, validation and test functions ##############################

### Define how to perform training
def training_loop(dataset, batch_size, net, optimizer, loss_fn, device, regularization=None):
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    batch_loss = []
    batch_acc = []

    for data, labels in train_loader:
      
      data = data.to(device).swapaxes(1, 0)
      labels = labels.to(device)

      net.train()
      #spk_rec, _, _, hid_rec = net(data)
      spk_rec, hid_rec = net(data)

      # Training loss
      if regularization != None:
        # L1 loss on spikes per neuron from the hidden layer
        reg_loss = regularization[0]*torch.mean(torch.sum(hid_rec, 0))
        # L2 loss on total number of spikes from the hidden layer
        reg_loss = reg_loss + regularization[1]*torch.mean(torch.sum(torch.sum(hid_rec, dim=0), dim=1)**2)
        loss_val = loss_fn(spk_rec, labels) + reg_loss
      else:
        loss_val = loss_fn(spk_rec, labels)

      batch_loss.append(loss_val.detach().cpu().item())

      # Training accuracy
      act_total_out = torch.sum(spk_rec, 0)  # sum over time
      _, neuron_max_act_total_out = torch.max(act_total_out, 1)  # argmax over output units to compare to labels
      batch_acc.extend((neuron_max_act_total_out == labels).detach().cpu().numpy()) 

      # Gradient calculation + weight update
      optimizer.zero_grad()
      loss_val.backward()
      optimizer.step()

    epoch_loss = np.mean(batch_loss)
    epoch_acc = np.mean(batch_acc)
    
    return [epoch_loss, epoch_acc]

### Define how to perform validation and test
def val_test_loop(dataset, batch_size, net, loss_fn, device, shuffle=True, saved_state_dict=None, label_probabilities=False, regularization=None):
  
  with torch.no_grad():
    if saved_state_dict != None:
        net.load_state_dict(saved_state_dict)
    net.to(device)
    net.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    batch_loss = []
    batch_acc = []

    for data, labels in loader:
        data = data.to(device).swapaxes(1, 0)
        labels = labels.to(device)

        #spk_out, _, _, hid_rec = net(data)
        spk_out, hid_rec = net(data)

        # Validation loss
        if regularization != None:
            # L1 loss on spikes per neuron from the hidden layer
            reg_loss = regularization[0]*torch.mean(torch.sum(hid_rec, 0))
            # L2 loss on total number of spikes from the hidden layer
            reg_loss = reg_loss + regularization[1]*torch.mean(torch.sum(torch.sum(hid_rec, dim=0), dim=1)**2)
            loss_val = loss_fn(spk_out, labels) + reg_loss
        else:
            loss_val = loss_fn(spk_out, labels)

        batch_loss.append(loss_val.detach().cpu().item())

        # Accuracy
        act_total_out = torch.sum(spk_out, 0)  # sum over time
        _, neuron_max_act_total_out = torch.max(act_total_out, 1)  # argmax over output units to compare to labels
        batch_acc.extend((neuron_max_act_total_out == labels).detach().cpu().numpy())
    
    if label_probabilities:
        log_softmax_fn = nn.LogSoftmax(dim=-1)
        log_p_y = log_softmax_fn(act_total_out)
        return [np.mean(batch_loss), np.mean(batch_acc)], torch.exp(log_p_y)
    else:
        return [np.mean(batch_loss), np.mean(batch_acc)]

################################################################################


###### 2) Log file configuration ###############################################

LOG = logging.getLogger(experiment_name)
LOG.setLevel(logging.DEBUG)
#LOG.debug("------------------------------------------------------------------------------------")
LOG.debug("Experiment started on: {}-{}-{} {}:{}:{}\n".format(
    experiment_datetime[:4],
    experiment_datetime[4:6],
    experiment_datetime[6:8],
    experiment_datetime[-6:-4],
    experiment_datetime[-4:-2],
    experiment_datetime[-2:])
    )

################################################################################


###### 4) Load data ############################################################

def NNI_load_data(settings, device):

    dataDir = os.path.join(parentDir,"data")

    ### Load the test subset (always the same)
    ds_test = torch.load(os.path.join(dataDir,"dataset_splits/{}/braille_letters_digits_40Hz_augmented_ds_test.pt".format(settings["experiment_data"])), map_location=device)

    ### Select random training and validation set
    rnd_idx = np.random.randint(0, 10)
    LOG.debug("Split number {} used for this experiment.\n".format(rnd_idx))
    ds_train = torch.load(os.path.join(dataDir,"dataset_splits/{}/braille_letters_digits_40Hz_augmented_ds_train_{}.pt".format(settings["experiment_data"],rnd_idx)), map_location=device)
    ds_val = torch.load(os.path.join(dataDir,"dataset_splits/{}/braille_letters_digits_40Hz_augmented_ds_val_{}.pt".format(settings["experiment_data"],rnd_idx)), map_location=device)

    return ds_train, ds_val, ds_test

letter_written = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


################################################################################


###### 5) Build the network ####################################################

def NNI_build(settings, input_size, num_steps, device):

    ### Network structure (input data --> encoding -> hidden -> output)

    input_channels = input_size #next(iter(ds_test))[0].shape[1]
    if settings["encoding_populations_needed"]:
        pop_size = int(settings["neurons_per_pop"]) # --> the number of neurons for the encoding layer with populations will be: pop_size*input_channels
        output_pop = int(settings["output_pop"])
        output_enc = output_pop*input_channels 
    else:
        pop_size = int(settings["neurons_per_pop"])
        output_enc = pop_size*input_channels 
    num_hidden = int(settings["nb_hidden"])
    num_outputs = len(load_Braille_dataset(settings["Braille_path"])[2])

    ### Surrogate gradient setting
    spike_grad = surrogate.fast_sigmoid(slope=int(settings["slope"]))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()

            ##### Initialize layers #####
            if settings["encoding_populations_needed"]:
                ### Encoding layer with populations
                self.pop_size = pop_size
                self.enc_pops = []
                self.lif_enc_pops = []
                for ii in range(input_channels):
                    self.enc_pops.append(nn.Linear(pop_size, output_pop).to(device))
                    self.lif_enc_pops.append(snn.Leaky(beta=settings["beta_enc"], learn_beta=True, learn_threshold=True, spike_grad=spike_grad).to(device))
                self.enc_pops = nn.ModuleList(self.enc_pops).to(device)
                self.lif_enc_pops = nn.ModuleList(self.lif_enc_pops).to(device)
            else:
                ### Encoding layer
                self.enc = nn.Linear(input_channels, output_enc).to(device)
                self.lif_enc = snn.Leaky(beta=settings["beta_enc"], learn_beta=True, learn_threshold=True, spike_grad=spike_grad).to(device)
            ### Recurrent layer
            self.fc1 = nn.Linear(output_enc, num_hidden).to(device)
            self.lif1 = snn.RLeaky(beta=settings["beta_hid"], learn_beta=True, learn_threshold=True, linear_features=num_hidden, spike_grad=spike_grad, reset_mechanism="zero").to(device)
            ### Output layer
            self.fc2 = nn.Linear(num_hidden, num_outputs).to(device)
            self.lif_enc = snn.Leaky(beta=settings["beta_enc"], learn_beta=True, learn_threshold=True, spike_grad=spike_grad, reset_mechanism="zero").to(device)

        def forward(self, x):

            ##### Initialize hidden states at t=0 #####
            if settings["encoding_populations_needed"]:
                ### Encoding layer with populations
                mem_pops_enc = torch.empty((input_channels,x.shape[1],output_pop), dtype=torch.float, device=device)
                spk_pops_enc = torch.empty((input_channels,x.shape[1],output_pop), dtype=torch.float, device=device)
                cur_pops_enc = torch.empty((input_channels,x.shape[1],output_pop), dtype=torch.float, device=device)
            else:
                ### Encoding layer
                mem_enc = self.lif_enc.init_leaky()
            ### Recurrent layer
            spk1, mem1 = self.lif1.init_rleaky()
            ### Output layer
            mem2 = self.lif2.init_leaky()

            # Record the spikes from the hidden layer
            spk1_rec = []
            # Record the final layer
            spk2_rec = []

            for step in range(num_steps):
                if settings["encoding_populations_needed"]:
                    ### Encoding layer with populations
                    for num,el in enumerate(self.enc_pops):
                        cur_pops_enc[num] = el(torch.tile(x[step,:,num],(self.pop_size,1)).swapaxes(1,0))
                    for num,el in enumerate(self.lif_enc_pops):
                        spk_pops_enc[num], mem_pops_enc[num] = el(cur_pops_enc[num], mem_pops_enc[num])
                    spk_enc = spk_pops_enc.clone().permute(1, 0, 2).reshape((x.shape[1],input_channels*output_pop)).requires_grad_(True)
                else:
                    ### Encoding layer
                    cur_enc = self.enc(x[step])
                    spk_enc, mem_enc = self.lif_enc(cur_enc, mem_enc)
                ### Recurrent layer
                cur1 = self.fc1(spk_enc) # self.fc1(x[step])
                spk1, mem1 = self.lif1(cur1, spk1, mem1)
                ### Output layer
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)

                spk1_rec.append(spk1)
                spk2_rec.append(spk2)

            return torch.stack(spk2_rec, dim=0), torch.stack(spk1_rec, dim=0)

    return Net().to(device)

################################################################################


###### 6) Set the simulation ###################################################

def NNI_run(settings, device):

    ### Load the data
    ds_train, ds_val, ds_test = NNI_load_data(settings, device)

    input_size = next(iter(ds_test))[0].shape[1]
    num_steps = next(iter(ds_test))[0].shape[0]

    ### Place the network onto the selected device        
    net = NNI_build(settings, input_size, num_steps, device)

    ### Set the loss function
    loss_fn = SF.ce_count_loss()
    regularization = [settings["reg_l1"], settings["reg_l2"]]

    ### Set the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=settings["lr"], betas=(0.9, 0.999))

    batch_size = settings["batch_size"]

    num_epochs = settings["nb_epochs"] # maximum number of epochs for training

    training_results = []
    validation_results = []

    ### Training and validation with a while loop including early stopping
    epoch = 0
    EarlyStop_delta_val_loss = 0.5 # intended as the percentage of change in validation loss
    counter_small_delta_loss = 0
    stop_small_delta_loss = 5 # how many times the condition must be met to induce early stopping
    EarlyStop_delta_val_loss_up = 0.5 # intended as the percentage of increase in validation loss
    counter_delta_loss_up = 0
    stop_delta_loss_up = 5 # how many times the condition must be met to induce early stopping
    EarlyStop_delta_val_acc_low = 0.1 # intended as the percentage of change in validation accuracy
    counter_small_delta_acc = 0
    stop_small_delta_acc = 5 # how many times the condition must be met to induce early stopping
    EarlyStop_delta_val_acc_high = 2 # intended as the percentage of decrease in validation accuracy
    counter_large_delta_acc = 0
    stop_large_delta_acc = 5 # how many times the condition must be met to induce early stopping

    while (counter_small_delta_loss < stop_small_delta_loss) & (counter_delta_loss_up < stop_delta_loss_up) & (counter_small_delta_acc < stop_small_delta_acc) & (counter_large_delta_acc < stop_large_delta_acc) & (epoch < num_epochs):
        
        epoch += 1

        train_loss, train_acc = training_loop(ds_train, batch_size, net, optimizer, loss_fn, device, regularization=regularization)
        val_loss, val_acc = val_test_loop(ds_val, batch_size, net, loss_fn, device, regularization=regularization)

        training_results.append([train_loss, train_acc])
        validation_results.append([val_loss, val_acc])

        LOG.debug("Epoch {}/{}: \n\ttraining loss: {} \n\tvalidation loss: {} \n\ttraining accuracy: {}% \n\tvalidation accuracy: {}%".format(epoch, num_epochs, training_results[-1][0], validation_results[-1][0], np.round(training_results[-1][1]*100,4), np.round(validation_results[-1][1]*100,4)))

        nni.report_intermediate_result({"default": np.round(val_acc*100,4),
                                        "training acc.": np.round(train_acc*100,4),
                                        "val. loss": np.round(val_loss,5),
                                        "train. loss": np.round(train_loss,5)})
        
        if val_acc >= np.max(np.array(validation_results)[:,1]):
            best_val_layers = copy.deepcopy(net.state_dict())
        
        if epoch >= 2:
            # Check loss variations (on validation data) during training: count number of epoch with SMALL (< EarlyStop_delta_val_loss %) CHANGES
            if np.abs(validation_results[-1][0] - validation_results[-2][0])/validation_results[-2][0]*100 < EarlyStop_delta_val_loss:
                counter_small_delta_loss += 1
            else:
                counter_small_delta_loss = 0
            # Check loss variations (on validation data) during training: count number of epoch with LARGE (> EarlyStop_delta_val_loss_up %) INCREASE
            if (validation_results[-1][0] - validation_results[-2][0])/validation_results[-2][0]*100 > EarlyStop_delta_val_loss_up:
                counter_delta_loss_up += 1
            else:
                counter_delta_loss_up = 0
            # check accuracy variations (on validation data) during training: count number of epoch with SMALL (> EarlyStop_delta_val_acc %) CHANGES
            if np.abs(validation_results[-1][1] - validation_results[-2][1])/validation_results[-2][1]*100 < EarlyStop_delta_val_acc_low:
                counter_small_delta_acc += 1
            else:
                counter_small_delta_acc = 0
            # check accuracy variations (on validation data) during training: count number of epoch with LARGE (> EarlyStop_delta_val_acc %) DECREASE
            if (validation_results[-2][1] - validation_results[-1][1])/validation_results[-2][1]*100 > EarlyStop_delta_val_acc_high:
                counter_large_delta_acc += 1
            else:
                counter_large_delta_acc = 0
    
    LOG.debug("\n")
    if counter_small_delta_loss >= stop_small_delta_loss:
        LOG.debug("Training stopped after {}/{} epochs: stop condition for small validation loss changes met.".format(epoch,num_epochs))
    elif counter_delta_loss_up >= stop_delta_loss_up:
        LOG.debug("Training stopped after {}/{} epochs: stop condition for validation loss increase met.".format(epoch,num_epochs))
    elif counter_small_delta_acc >= stop_small_delta_acc:
        LOG.debug("Training stopped after {}/{} epochs: stop condition for small validation accuracy changes met.".format(epoch,num_epochs))
    elif counter_large_delta_acc >= stop_large_delta_acc:
        LOG.debug("Training stopped after {}/{} epochs: stop condition for validation accuracy decrease met.".format(epoch,num_epochs))
    else:
        LOG.debug("Training ended after {}/{} epochs.".format(epoch,num_epochs))

    training_hist = np.array(training_results)
    validation_hist = np.array(validation_results)

    # best training and validation at best training
    acc_best_train = np.max(training_hist[:,1])
    epoch_best_train = np.argmax(training_hist[:,1])
    acc_val_at_best_train = validation_hist[epoch_best_train][1]

    # best validation and training at best validation
    acc_best_val = np.max(validation_hist[:,1])
    epoch_best_val = np.argmax(validation_hist[:,1])
    acc_train_at_best_val = training_hist[epoch_best_val][1]

    LOG.debug("\n")
    LOG.debug("Trial results: ")
    LOG.debug("\tBest training accuracy: {}% ({}% corresponding validation accuracy) at epoch {}/{}".format(
        np.round(acc_best_train*100,4), np.round(acc_val_at_best_train*100,4), epoch_best_train+1, num_epochs))
    LOG.debug("\tBest validation accuracy: {}% ({}% corresponding training accuracy) at epoch {}/{}".format(
        np.round(acc_best_val*100,4), np.round(acc_train_at_best_val*100,4), epoch_best_val+1, num_epochs))
    LOG.debug("\n")
    
    test_acc_N = []

    for ii in range(settings["n_test"]):

        test_results = val_test_loop(ds_test, batch_size, net, loss_fn, device, shuffle=False, saved_state_dict=best_val_layers, regularization=regularization)
        
        test_acc_N.append(test_results[1])
        
        LOG.debug("Test {}/{}: {}%".format(ii+1,settings["n_test"],np.round(test_results[1]*100,4)))

    LOG.debug("\n")
    LOG.debug("Min. test accuracy: {}%".format(np.round(np.min(test_acc_N)*100,4)))
    LOG.debug("Max. test accuracy: {}%".format(np.round(np.max(test_acc_N)*100,4)))
    LOG.debug("Mean test accuracy: {}%".format(np.round(np.mean(test_acc_N)*100,4)))
    LOG.debug("Median test accuracy: {}%".format(np.round(np.median(test_acc_N)*100,4)))
    LOG.debug("Std. Dev. test accuracy: {}%\n".format(np.round(np.std(test_acc_N)*100,4)))

    # Ns single-sample inferences to check label probbailities
    Ns = 10
    for ii in range(Ns):
        single_sample = next(iter(DataLoader(ds_test, batch_size=1, shuffle=True)))
        _, lbl_probs = val_test_loop(TensorDataset(single_sample[0],single_sample[1]), 1, net, loss_fn, device, shuffle=False, saved_state_dict=best_val_layers, label_probabilities=True, regularization=regularization)
        LOG.debug("Single-sample inference {}/{} from test set:".format(ii+1,Ns))
        LOG.debug("Sample: {} \tPrediction: {}".format(letter_written[single_sample[1]],letter_written[torch.max(lbl_probs.cpu(),1)[1]]))
        LOG.debug("Label probabilities (%): {}".format(np.round(np.array(lbl_probs.detach().cpu().numpy())*100,2)))

    LOG.debug("------------------------------------------------------------------------------------\n\n")

    nni.report_final_result({"default": np.round(acc_best_val*100,4), # the default value is the maximum validation accuracy achieved
                             "best training": np.round(acc_best_train*100,4),
                             "median test": np.round(np.median(test_acc_N)*100,4),
                             "std. dev. test": np.round(np.std(test_acc_N)*100,4)})

    return training_results, validation_results, test_results, test_acc_N, best_val_layers

################################################################################


###### 7) RUN THE NNI EXPERIMENT ###############################################

try:

    trial_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    LOG.debug("\n")
    LOG.debug("Trial {} (# {}, ID {}) started on: {}-{}-{} {}:{}:{}\n".format(nni.get_sequence_id()+1,
                                                                                nni.get_sequence_id(),
                                                                                nni.get_trial_id(),
                                                                                trial_datetime[:4],
                                                                                trial_datetime[4:6],
                                                                                trial_datetime[6:8],
                                                                                trial_datetime[-6:-4],
                                                                                trial_datetime[-4:-2],
                                                                                trial_datetime[-2:]))

    ### Every n_tr, "update" the searchspace inducing a new RandomState for the tuner
    n_tr = 200
    if (nni.get_sequence_id() > 0) & (nni.get_sequence_id()%n_tr == 0):
        update_searchspace = SearchSpaceUpdater({"filename": searchspace_path, "id": nni.get_experiment_id()})
        updater.update_searchspace(update_searchspace) # it will use update_searchspace.filename to update the search space

    ### Get parameters from the tuner combining them with the line arguments
    settings_nni = nni.get_next_parameter()
    for ii in settings_nni.keys():
        if ii in settings.keys():
            del settings[ii]
    
    PARAMS = {**settings, **settings_nni}

    LOG.debug("Parameters selected for trial {} (# {}, ID {}): {}\n".format(
        nni.get_sequence_id()+1, nni.get_sequence_id(), nni.get_trial_id(), PARAMS))
    
    training_results, validation_results, test_results, test_acc_N, best_val_layers = NNI_run(PARAMS, device)

    ### Report results (i.e. test accuracy from best validation) of each trial
    path = './results/reports/{}'.format(experiment_name)
    create_directory(path)
    report_path = path + "/{}".format(nni.get_experiment_id())
    with open(report_path, 'a') as f:
        f.write("{} % \t median test accuracy ({}) from {} repetitions".format(np.median(test_acc_N)*100,nni.get_trial_id(),settings["n_test"]))
        f.write('\n')
    
    ### Save trained weights giving the highest (median) test accuracy
    if settings["save_weights"]:
        path = './results/layers/{}'.format(experiment_name)
        create_directory(path)
        save_layers_path = path + "/{}.pt".format(nni.get_experiment_id())
        with open(report_path, 'r') as f:
            if np.median(test_acc_N)*100 >= np.max(np.asarray([(line.strip().split(" ")[0]) for line in f], dtype=np.float64)):
                torch.save(best_val_layers, save_layers_path)
                if settings["store_weights"]: # might be redundant but hopefully safer than just saving with NNI-based names
                    test_ref = float(np.round(np.median(test_acc_N)*100,2))
                    if ["layers_ref_" in ii for ii in os.listdir(path)]:
                        del_list = []
                        for ii in os.listdir(path):
                            if (".pt" in ii) & ("layers_ref_" in ii):
                                if float(ii.split("layers_ref_")[1].split(".pt")[0].split("_")[0]) < 0.5*test_ref:
                                    del_list.append(ii) # weights saved from models with less than 50% in performance compared to the top are deleted
                        for ii in del_list:
                            os.remove(os.path.join(path,ii))
                    store_layers_path = path + "/layers_ref_{}_{}.pt".format(test_ref,experiment_datetime)
                    torch.save(best_val_layers, store_layers_path)

except Exception as e:
    LOG.exception(e)
    raise

################################################################################
