"""
engine_vae.py

Derived engine class for training a unsupervised generative VAE
"""

# WatChMaL imports
from training_utils.engine import Engine
from training_utils.loss_funcs import VAELoss

# Global variables
_LOG_KEYS = ["loss", "recon_loss", "kl_loss"]
_DUMP_KEYS = ["recon", "z", "mu", "logvar"]

class EngineVAE(Engine):
    
    def __init__(self, model, config):
        super().__init__(model, config)
        self.criterion = VAELoss
        
        # Split the dataset into labelled and unlabelled subsets
        # Note : Only the unlabelled subset will be used for VAE training
        n_cl_train = int(len(self.dset.train_indices) * config.cl_ratio)
        n_cl_val   = int(len(self.dset.val_indices) * config.cl_ratio)
        
        self.train_indices = self.dset.train_indices[n_cl_train:]
        self.val_indices   = self.dset.val_indices[n_cl_val:]
        self.test_indices  = self.dset.test_indices
        
        # Initialize the torch dataloaders
        self.train_loader = DataLoader(self.dset, batch_size=config.batch_size_train, shuffle=False,
                                       pin_memory=True, sampler=SubsetRandomSampler(self.train_indices))
        self.val_loader   = DataLoader(self.dset, batch_size=config.batch_size_val, shuffle=False,
                                       pin_memory=True, sampler=SubsetRandomSampler(self.val_indices))
        self.test_loader  = DataLoader(self.dset, batch_size=config.batch_size_test, shuffle=False,
                                       pin_memory=True, sampler=SubsetRandomSampler(self.test_indices))
        
        # Define the placeholder attributes
        self.data           = None
        self.labels         = None
        self.energies       = None
        
        # Attributes to allow beta annealing of the ELBO Loss
        self.iteration      = None
        self.num_iteartions = None
        
    def forward(self, mode):
        """Overrides the forward abstract method in Engine.py.
        
        Args:
        mode -- One of 'train', 'validation' to set the correct grad_mode
        """
        
        ret_dict = None
        
        if self.data is not None and len(self.data.size()) == 4 and mode in ["train", "validation"]:
            self.data = self.data.to(self.device)
            self.data = self.data.permute(0,3,1,2)
            
        # Set the correct grad_mode given the mode
        if mode == "train":
            grad_mode = True
            self.model.train()
        elif mode in ["validation", "sample", "decode"]:
            grad_mode= False
            self.model.eval()
            
        if mode in ["train", "validation"]:
            recon, z, mu, logvar      = self.model(self.data, mode)
            loss, recon_loss, kl_loss = self.criterion(recon, data, mu, logvar, self.iteration, self.num_iterations)
            self.loss                 = loss
            
            ret_dict                  = {"loss"       : loss.cpu().detach().item(),
                                         "recon_loss" : recon_loss.cpu().detach().item(),
                                         "kl_loss"    : kl_loss.cpu().detach().item(),
                                         "recon"      : recon.cpu().detach().numpy(),
                                         "z"          : z.cpu().detach().numpy(),
                                         "mu"         : mu.cpu().detach().numpy(),
                                         "logvar"     : logvar.cpu().detach().numpy()}
        elif mode == "sample":
            recon, z = self.model(self.data, mode)
            ret_dict = {"recon" : recon.cpu().detach().numpy(),
                        "z"     : z.cpu().detach().numpy()}
        elif mode == "decode":
            recon    = self.model(self.data, mode)
            ret_dict = {"recon" : recon.cpu().detach().numpy()}
        
        if self.data is not None and len(self.data.size()) == 4 and mode in ["train", "validation"]:
            self.data = self.data.permute(0,2,3,1)
            
        return ret_dict
    
    def train(self, epochs, report_interval, num_vals, num_val_batches):
       """Overrides the train method in Engine.py.
        
        Args:
        epcohs          -- Number of epochs to train the model for
        report_interval -- Interval at which to report the training metrics to the user
        num_vals        -- Number of validations to perform throughout training
        num_val_batches -- Number of batches to use during each validation
        """
        # Initialize the iterator over the validation subset
        val_iter = iter(self.val_loader)
        
        # Set the iterations at which to dump the events and their metrics
        dump_interations = self.dump_iterations(num_vals)
        
        # Initialize epoch counter
        epoch = 0.
        
        # Initialize iteration counter
        iteration = 0
        
        # Parameter to upadte when saving the best model
        best_loss = 1000000.
        
        # Global training loop for multiple epochs
        while (floor(epoch) < epochs):
            
            print('Epoch',floor(epoch),
                  'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))
        
            # Local training loop for a single epoch
            for i, data in enumerate(self.train_loader):
                
                # Using only the charge data [:19]
                self.data     = data[0][:,:,:,:19].float()
                self.labels   = data[1].long()
                self.energies = data[2]

                # Do a forward pass using data = self.data
                res = self.forward(mode="train")
                
                # Do a backward pass using loss = self.loss
                self.backward()
                
                # Update the epoch and iteration
                epoch     += 1./len(self.train_loader)
                iteration += 1
                
                # Iterate over the _LOG_KEYS and add the vakues to a list
                keys   = ["iteration", "epoch"]
                values = [iteration, epoch]
                
                for key in _LOG_KEYS:
                    if key in res.keys():
                        keys.append(key)
                        values.append(res[key])
                        
                # Record the metrics for the mini-batch in the log
                self.train_log.record(keys, values)
                self.train_log.write()
                
                # Print the metrics at given intervals
                if i == 0 or (i+1)%report_interval == 0:
                    print("... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Accuracy %1.3f" %
                          (iteration, epoch, res["loss"], res["accuracy"]))
                    
                # Run validation on given intervals
                if iteration%valid_interval == 0:
                    
                    curr_loss = 0.
                    val_batch = 0
                    keys = ['iteration','epoch']
                    values = [iteration, epoch]
                    
                    local_values = []
                    
                    for val_batch in range(num_val_batches):
                        
                        try:
                            val_data = next(val_iter)
                        except StopIteration:
                            val_iter = iter(self.val_iter)
                        
                        # Extract the event data from the input data tuple
                        self.data     = val_data[0][:,:,:,:19].float()
                        self.labels   = val_data[1].long()
                        self.energies = val_data[2].float()

                        res = self.forward(mode="validation")
                        
                        if val_batch == 0:
                            for key in _LOG_KEYS:
                                if key in res.keys():
                                    keys.append(key)
                                    local_values.append(res[key])
                        else:
                            log_index = 0
                            for key in _LOG_KEYS:
                                if key in res.keys():
                                    local_values[log_index] += res[key]
                                    log_index += 1
                                    
                        curr_loss += res["loss"]

                    for local_value in local_values:
                        values.append(local_value/num_val_batches)
                    
                    # Record the validation stats to the csv
                    self.val_log.record(keys,values)
                    
                    # Average the loss over the validation batch
                    curr_loss = curr_loss / num_val_batches

                    if iteration in dump_iterations:
                        save_arr_keys = ["events", "labels", "energies"]
                        save_arr_values = [self.data.cpu().numpy(), val_data[1], val_data[2]]
                        for key in _DUMP_KEYS:
                            if key in res.keys():
                                save_arr_keys.append(key)
                                save_arr_values.append(res[key])

                        # Save the actual and reconstructed event to the disk
                        savez(self.dirpath + "/iteration_" + str(iteration) + ".npz",
                              **{key:value for key,value in zip(save_arr_keys,save_arr_values)})

                    # Save the best model
                    if curr_loss < best_loss:
                        self.save_state(mode="best")
                    
                    # Save the latest model
                    self.save_state(mode="latest")

                    self.val_log.write()
                    
                if epoch >= epochs:
                    break
                    
            print('... Iteration %d ... Epoch %1.2f ... Loss %1.3f' % (iteration, epoch, res['loss']))
            
        self.val_log.close()
        self.train_log.close()
    
    
        
    
        
        
        
        
        
        
        
    
    