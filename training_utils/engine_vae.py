"""
engine_vae.py

Engine to train, validate and test various VAE architectures

Author  : Abhishek .
"""

# PyTorch imports
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Standard and data processing imports
import os
import sys
import time
import numpy as np

# WatChMaL imports
from io_utils import ioconfig
from io_utils.data_handling import WCH5Dataset
from plot_utils.notebook_utils import CSVData

# Class for the training engine for the WatChMaLVAE
class EngineVAE:
    
    """
    Purpose : Training engine for the WatChMaLVAE. Performs training, validation,
              and testing of the models
    """
    def __init__(self, model, config):
        self.model = model
        if (config.device == 'gpu') and config.gpu_list:
            print("Requesting GPUs. GPU list : " + str(config.gpu_list))
            self.devids = ["cuda:{0}".format(x) for x in config.gpu_list]

            print("Main GPU: "+self.devids[0])
            if torch.cuda.is_available():
                self.device = torch.device(self.devids[0])
                if len(self.devids) > 1:
                    print("Using DataParallel on these devices: {}".format(self.devids))
                    self.model = nn.DataParallel(self.model, device_ids=config.gpu_list, dim=0)

                print("CUDA is available")
            else:
                self.device=torch.device("cpu")
                print("CUDA is not available")
        else:
            print("Unable to use GPU")
            self.device=torch.device("cpu")

        self.model.to(self.device)

        # Initialize the optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.0001)
        self.criterion = self.VAELoss
        self.recon_loss = nn.MSELoss()

        #placeholders for data and labels
        self.data=None
        self.labels=None
        self.iteration=None

        # NOTE: The functionality of this block is coupled to the implementation of WCH5Dataset in the iotools module
        self.dset=WCH5Dataset(config.path,
                              config.val_split,
                              config.test_split,
                              shuffle=config.shuffle,
                              reduced_dataset_size=config.subset)

        self.train_iter=DataLoader(self.dset,
                                   batch_size=config.batch_size_train,
                                   shuffle=False,
                                   sampler=SubsetRandomSampler(self.dset.train_indices))
        
        self.val_iter=DataLoader(self.dset,
                                 batch_size=config.batch_size_val,
                                 shuffle=False,
                                 sampler=SubsetRandomSampler(self.dset.val_indices))
        
        self.test_iter=DataLoader(self.dset,
                                  batch_size=config.batch_size_test,
                                  shuffle=False,
                                  sampler=SubsetRandomSampler(self.dset.test_indices))

        self.dirpath=config.dump_path + time.strftime("%Y%m%d_%H%M%S") + "/"
        
        try:
            os.stat(self.dirpath)
        except:
            print("Creating a directory for run dump: {}".format(self.dirpath))
            os.mkdir(self.dirpath)

        self.config=config
        
        # Save a copy of the config in the dump path
        ioconfig.saveConfig(self.config, self.dirpath + "config_file.ini")
        
    # Loss function for the VAE combining MSELoss and KL-divergence
    def VAELoss(self, reconstruction, mean, log_var, data):
        
        # MSE Reconstruction Loss
        mse_loss = self.recon_loss(reconstruction, data)
        
        # KL-divergence Loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        
        return mse_loss + kl_loss, mse_loss, kl_loss
    
        
    # Method to compute the loss using the forward pass
    def forward(self, mode="train"):
        
        # Move the data to the user-specified device
        self.data = self.data.to(self.device)
        self.data = self.data.permute(0,3,1,2)
        
        if mode == "train" or mode == "validate":
            
            grad_mode = True if mode == "train" else False
            
            with torch.set_grad_enabled(grad_mode):

                # Collect the output from the model
                z, prediction, mu, logvar = self.model(self.data, mode)
                # Training
                loss = -1
                loss, mse_loss, kl_loss = self.criterion(prediction, mu, logvar, self.data)
                self.loss = loss

                # Restore the shape of the data and the prediction
                self.data = self.data.permute(0,2,3,1)
                prediction = prediction.permute(0,2,3,1)

            return {"loss"       : loss.cpu().detach().item(),
                    "mse_loss"   : mse_loss.cpu().detach().item(),
                    "kl_loss"    : kl_loss.cpu().detach().item(),
                    "z"          : z.cpu().detach().numpy(),
                    "prediction" : prediction.cpu().detach().numpy(),
                    "mu"         : mu.cpu().detach().numpy(),
                    "logvar"      : logvar.cpu().detach().numpy()}
        
        elif mode == "generate":
            
            with torch.set_grad_enabled(False):
                
                # Call the model to generate the latent vectors
                z_gen, _, _ = self.model(self.data, mode)
                # Extract the latent vector
                z_gen.cpu().detach().numpy()
                
                # Restore the shape of the data
                self.data = self.data.permute(0,2,3,1)
                
                return {"z_gen" : z_gen.cpu().detach().numpy()}
            
        elif mode == "sample":
            
            pass
        
    def backward(self):
        
        self.optimizer.zero_grad()  # Reset gradient accumulation
        self.loss.backward()        # Propagate the loss backwards
        self.optimizer.step()       # Update the optimizer parameters
        
    def train(self, epochs=10.0, report_interval=10, valid_interval=1000):

        # Prepare attributes for data logging
        self.train_log = CSVData(self.dirpath+'log_train.csv')
        self.val_log = CSVData(self.dirpath+'val_test.csv')
        
        # Variables to save the actual and reconstructed events
        np_event_path = self.dirpath+"/iteration_"
        
        # Set neural net to training mode
        self.model.train()
        
        # Initialize epoch counter
        epoch = 0.
        
        # Initialize iteration counter
        iteration = 0
        
        # Parameter to save the best model
        best_loss = 1000000.0
        
        # Training loop
        while (int(epoch+0.5) < epochs):
            
            print('Epoch',int(epoch+0.5),
                  'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            
            # Loop over data samples and into the network forward function
            for i, data in enumerate(self.train_iter):
                
                # Move the data to the device specified by the user
                self.data = data[0][:,:,:,:19]
                
                # Call forward: make a prediction & measure the average error
                res = self.forward(mode="train")
                
                # Call backward: backpropagate error and update weights
                self.backward()
                
                # Epoch update
                epoch += 1./len(self.train_iter)
                iteration += 1
                
                # Log/Report
                # Record the current performance on train set
                self.train_log.record(['iteration','epoch','loss', 'mse_loss', 'kl_loss'],
                                      [iteration, epoch, res['loss'], res['mse_loss'],
                                       res['kl_loss']])
                self.train_log.write()
                
                # once in a while, report
                if i==0 or (i+1)%report_interval == 0:
                    print('... Iteration %d ... Epoch %1.2f ... Loss %1.3f' % 
                          (iteration, epoch, res['loss']))
                    
                # Run validation on user-defined intervals
                if (iteration+1)%valid_interval == 0:
                        
                    self.model.eval()
                    val_data = next(iter(self.val_iter))

                    # Extract the event data from the input data tuple
                    self.data = val_data[0][:,:,:,:19]

                    res = self.forward(mode="validate")

                    # Save the actual and reconstructed event to the disk
                    np.savez(np_event_path + str(iteration) + ".npz",
                             events=self.data.cpu().numpy(), z=res['z'], recons=res['prediction'],
                             mus=res["mu"], logvars=res["logvar"], labels=val_data[1],
                             energies=val_data[3])

                    # Record the validation stats to the csv
                    self.val_log.record(['iteration','epoch','loss', 'mse_loss', 'kl_loss'],
                                      [iteration, epoch, res['loss'], res['mse_loss'],
                                       res['kl_loss']])
                    
                    # Save the best model
                    if res['loss'] < best_loss:
                        self.save_state(model_type="best")
                    
                    # Save the best model
                    self.save_state(model_type="latest")

                    self.val_log.write()
                    self.model.train()
                    
                if epoch >= epochs:
                    break
                    
            print('... Iteration %d ... Epoch %1.2f ... Loss %1.3f' % (iteration, epoch, res['loss']))
            
        self.val_log.close()
        self.train_log.close()
        
    def sample(self, num_samples=10):
        
        # Setup the path
        sample_save_path = self.dirpath + 'samples/'
        
        # Check if the iteration has not been specified
        if self.iteration is None:
            self.iteration = 0
            
        # Create the directory if it does not already exist
        if not os.path.exists(sample_save_path):
            os.mkdir(sample_save_path)
        
        sample_save_path = sample_save_path + str(self.config.model[1]) + '_' + str(self.iteration)
        
        # Create the directory if it does not already exist
        if not os.path.exists(sample_save_path):
            os.mkdir(sample_save_path)
            
        # Samples list
        sample_list = []
        
        # Put the model in eval mode
        self.model.eval()
        
        # Iterate over the counter
        for i in range(num_samples):
            
            with torch.no_grad():

                _, sample = self.model(None, mode="sample")
                sample_list.extend(sample.permute(0,2,3,1).cpu().detach().numpy())
                
        # Put the model back in train mode
        self.model.train()
        
        # Convert the list to an numpy array and save to the given path
        np.save(sample_save_path + '/' + "{0}_samples".format(str(num_samples)) + ".npy", np.array(sample_list))
        
    # Generate and save the latent vectors for training and validation sets
    
    def generate_latent_vectors(self, mode="pre", report_interval=10):
        
        # Setup the save path for the vectors
        train_save_path = self.dirpath + mode + "_train_latent.npz"
        valid_save_path = self.dirpath + mode + "_valid_latent.npz"
        
        # Switch the model
        self.model.eval()
        
        # List to hold the values
        train_latent_list = []
        train_labels_list = []
        train_energies_list = []
        
        valid_latent_list = []
        valid_labels_list = []
        valid_energies_list = []
        
        # Print message
        print("Generating latent vectors over the training data")
        
        with torch.set_grad_enabled(False):
        
            # Iterate over the training samples
            for i, data in enumerate(self.train_iter):

                # once in a while, report
                if i==0 or (i+1)%report_interval == 0:
                    print('... Training data iteration %d ...' %(i))

                # Use only the charge data for the events
                self.data, labels, energies = data[0][:,:,:,:19],data[1], data[3]
                
                res = self.forward(mode="generate")

                # Add the values to the lists
                train_latent_list.extend(res['z_gen'])
                train_labels_list.extend(labels)
                train_energies_list.extend(energies)

            # Print message
            print("Generating latent vectors over the validation data")

            # Iterate over the validation samples
            for i, data in enumerate(self.val_iter):

                # once in a while, report
                if i==0 or (i+1)%report_interval == 0:
                    print('... Validation data iteration %d ...' %(i))

                # Use only the charge data for the events
                self.data, labels, energies = data[0][:,:,:,:19],data[1], data[3]

                res = self.forward(mode="generate")

                # Add the values to the lists
                valid_latent_list.extend(res['z_gen'])
                valid_labels_list.extend(labels)
                valid_energies_list.extend(energies)
            
        # Save the lists as numpy arrays
        np.savez(train_save_path,
                 latents=train_latent_list,
                 labels=train_labels_list,
                 energies=train_energies_list)
        
        np.savez(valid_save_path,
                 latents=valid_latent_list,
                 labels=valid_labels_list,
                 energies=valid_energies_list)
        
        # Switch the model back to training
        self.model.train()
        
        
    def save_state(self, model_type="latest"):
            
        filename = self.dirpath+"/"+str(self.config.model[1])+"_"+model_type+".pth"
        # Save parameters
        # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
        # 2) network weight
        torch.save({
            'global_step': self.iteration,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': self.model.state_dict()
        }, filename)
        return filename
    
    
    def restore_state(self, weight_file):
        
        weight_file = self.config.restore_state
        
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            
            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f)
            
            # load network weights
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            
            # if optim is provided, load the state of the optim
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                
            # load iteration count
            self.iteration = checkpoint['global_step']