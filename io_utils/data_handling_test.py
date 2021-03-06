"""
WCH5Dataset update to apply normalization on the fly to the test dataset
"""

# PyTorch imports
from torch.utils.data import Dataset
import h5py

import numpy as np
import numpy.ma as ma
import math
import random
import pdb

# WatChMaL imports
import preprocessing.normalize_funcs as norm_funcs

# Returns the maximum height at which cherenkov radiation will hit the tank
def find_bounds(pos, ang, label, energy):
    # Arguments:
    # pos - position of particles
    # ang - polar and azimuth angles of particle
    # label - type of particle
    # energy - particle energy
    
    '''
    #label = np.where(label==0, mass_dict[0], label)
    #label = np.where(label==1, mass_dict[1], label)
    #label = np.where(label==2, mass_dict[2], label)
    #beta = ((energy**2 - label**2)**0.5)/energy
    #max_ang = abs(np.arccos(1/(1.33*beta)))*1.5
    '''
    max_ang = abs(np.arccos(1/(1.33)))*1.05
    theta = ang[:,1]
    phi = ang[:,0]
    
    # Determine shortest distance emission will travel before it hits the tank
    # It checks the middle and edges of the emitted ring
    
    # radius of barrel
    r = 400
    
    # position of particle in barrel
    end = np.array([pos[:,0], pos[:,2]]).transpose()
    
    # Checks one edge of the ring
    # a point along the particle direction (plus max Cherenkov angle) located outside of the barrel
    start = end + 1000*(np.array([np.cos(theta + max_ang), np.sin(theta + max_ang)]).transpose())
    # finding intersection of particle with barrel
    a = (end[:,0] - start[:,0])**2 + (end[:,1] - start[:,1])**2
    b = 2*(end[:,0] - start[:,0])*(start[:,0]) + 2*(end[:,1] - start[:,1])*(start[:,1])
    c = start[:,0]**2 + start[:,1]**2 - r**2
    t = (-b - (b**2 - 4*a*c)**0.5)/(2*a)
    intersection = np.array([(end[:,0]-start[:,0])*t,(end[:,1]-start[:,1])*t]).transpose() + start
    length = end - intersection
    length1 = (length[:,0]**2 + length[:,1]**2)**0.5
    
    # Checks the middle of the ring
    # a point along the particle direction located outside of the barrel
    start = end + 1000*(np.array([np.cos(theta - max_ang), np.sin(theta - max_ang)]).transpose())
    # finding intersection of particle with barrel
    a = (end[:,0] - start[:,0])**2 + (end[:,1] - start[:,1])**2
    b = 2*(end[:,0] - start[:,0])*(start[:,0]) + 2*(end[:,1] - start[:,1])*(start[:,1])
    c = start[:,0]**2 + start[:,1]**2 - r**2
    t = (-b - (b**2 - 4*a*c)**0.5)/(2*a)
    intersection = np.array([(end[:,0]-start[:,0])*t,(end[:,1]-start[:,1])*t]).transpose() + start
    length = end - intersection
    length2 = (length[:,0]**2 + length[:,1]**2)**0.5 
    
    # Checks the other edge of the ring
    # a point along the particle direction (minus max Cherenkov angle) located outside of the barrel
    start = end + 1000*(np.array([np.cos(theta), np.sin(theta)]).transpose())
    # finding intersection of particle with barrel
    a = (end[:,0] - start[:,0])**2 + (end[:,1] - start[:,1])**2
    b = 2*(end[:,0] - start[:,0])*(start[:,0]) + 2*(end[:,1] - start[:,1])*(start[:,1])
    c = start[:,0]**2 + start[:,1]**2 - r**2
    t = (-b - (b**2 - 4*a*c)**0.5)/(2*a)
    intersection = np.array([(end[:,0]-start[:,0])*t,(end[:,1]-start[:,1])*t]).transpose() + start
    length = end - intersection
    length3 = (length[:,0]**2 + length[:,1]**2)**0.5 
    
    length = np.maximum(np.maximum(length1,length2), length3)

    top_ang = math.pi/2 - np.arctan((520 - pos[:,2])/ length)
    bot_ang = math.pi/2 + np.arctan(abs(-520 - pos[:,2])/length)
    lb = top_ang + max_ang
    ub = bot_ang - max_ang
    return np.array([lb, ub, np.minimum(np.minimum(length1,length2), length3)]).transpose()


class WCH5DatasetTest(Dataset):
    """
    Dataset storing image-like data from Water Cherenkov detector
    memory-maps the detector data from the hdf5 file
    The detector data must be uncompresses and unchunked
    labels are loaded into memory outright
    No other data is currently loaded
    """

    def __init__(self, test_dset_path, test_idx_path, norm_params_path, chrg_norm="identity", time_norm="identity", shuffle=1, test_subset=None, num_datasets=1,seed=42,label_map=None):
        
        assert hasattr(norm_funcs, chrg_norm) and hasattr(norm_funcs, time_norm), "Functions "+ chrg_norm + " and/or " + time_norm + " are not implemented in normalize_funcs.py, aborting."
        
        if label_map is not None:
            #make the fxn
            self.label_map = lambda x : label_map[1] if x==label_map[0] else x
        else:
            self.label_map = lambda x : x

        # Load the normalization parameters used by normalize_hdf5 methods
        norm_params = np.load(norm_params_path, allow_pickle=True)
        self.chrg_acc = norm_params["c_acc"]
        self.time_acc = norm_params["t_acc"]

        self.chrg_func = getattr(norm_funcs, chrg_norm)
        self.time_func = getattr(norm_funcs, time_norm)
        
        self.event_data = []
        self.labels = []
        self.energies = []
        self.positions = []
        self.angles = []
        
        self.train_indices = []
        self.val_indices = []
        
        
        self.event_data = []
        self.labels = []
        self.energies = []
        self.positions = []
        self.angles = []
        self.eventids = []
        self.rootfiles = []
        
        self.test_indices = []
        
        for i in np.arange(num_datasets):
            '''
            
            # Import test events from h5 file
            filtered_index = "/fast_scratch/WatChMaL/data/IWCD_fulltank_300_pe_idxs.npz"
            filtered_indices = np.load(filtered_index, allow_pickle=True)
            test_filtered_indices = filtered_indices['test_idxs']

            original_data_path = "/data/WatChMaL/data/IWCDmPMT_4pi_fulltank_9M.h5"
            f = h5py.File(original_data_path, "r")

            hdf5_event_data = (f["event_data"])
            self.event_data.append(np.memmap(original_data_path, mode="r", shape=hdf5_event_data.shape,
                                                offset=hdf5_event_data.id.get_offset(), dtype=hdf5_event_data.dtype))

            original_eventids = np.array(f['event_ids'])
            original_rootfiles = np.array(f['root_files'])
            original_energies = np.array(f['energies'])
            original_positions = np.array(f['positions'])
            original_angles = np.array(f['angles'])
            original_labels = np.array(f['labels'])

            filtered_eventids = original_eventids[test_filtered_indices]
            filtered_rootfiles = original_rootfiles[test_filtered_indices]
            filtered_energies = original_energies[test_filtered_indices]
            filtered_positions = original_positions[test_filtered_indices]
            filtered_angles = original_angles[test_filtered_indices]
            filtered_labels = original_labels[test_filtered_indices]
            
            self.labels.append(filtered_labels)
            self.energies.append(filtered_energies)
            self.positions.append(filtered_positions)
            self.angles.append(filtered_angles)
            self.eventids.append(filtered_eventids)
            self.rootfiles.append(filtered_rootfiles)
            
            
            self.reduced_size = test_subset
            '''
            f = h5py.File(test_dset_path[i], "r")

            hdf5_event_data = f["event_data"]
            hdf5_labels = f["labels"]
            hdf5_energies = f["energies"]
            hdf5_positions = f["positions"]
            hdf5_angles = f["angles"]
            hdf5_eventids = f["event_ids"]
            hdf5_rootfiles = f["root_files"]

            assert hdf5_event_data.shape[0] == hdf5_labels.shape[0]

            # Create a memory map for event_data - loads event data into memory only on __getitem__()
            self.event_data.append(np.memmap(test_dset_path[i], mode="r", shape=hdf5_event_data.shape,
                                        offset=hdf5_event_data.id.get_offset(), dtype=hdf5_event_data.dtype))

            # Load the contents which could fit easily into memory
            self.labels.append(np.array(hdf5_labels))
            self.energies.append(np.array(hdf5_energies))
            self.positions.append(np.array(hdf5_positions))
            self.angles.append(np.array(hdf5_angles))
            self.eventids.append(np.array(hdf5_eventids))
            self.rootfiles.append(np.array(hdf5_rootfiles))
            
            # Running only on events that went through fiTQun
            
            

            # Set the total size of the trainval dataset to use
            self.reduced_size = test_subset                
            
            
            if test_idx_path[i] is not None:

                test_indices = np.load(test_idx_path[i], allow_pickle=True)
                self.test_indices.append(test_indices["test_idxs"])
                self.test_indices[i] = self.test_indices[i][:]
                print("Loading test indices from: ", test_idx_path[i])
            
            else:
                
                test_indices = np.arange(self.labels[i].shape[0])
                np.random.shuffle(test_indices)
                #n_test = int(0.9 * test_indices.shape[0])
                #self.test_indices[i] = test_indices[n_test:]
                self.test_indices.append(test_indices)
                    
                
            #np.random.shuffle(self.test_indices[i])

            ## Seed the pseudo random number generator
            #if seed is not None:
                #np.random.seed(seed)

            # Shuffle the indices
            #if shuffle:
                #np.random.shuffle(self.test_indices[i])

            # If using a subset of the entire dataset
            if self.reduced_size is not None:
                assert len(self.test_indices[i])>=self.reduced_size
                self.test_indices[i] = np.random.choice(self.labels[i].shape[0], self.reduced_size)
            
            # DATA SLICING
            # For center dataset:
            '''
            # find barrel-only events
            #max_ang = abs(np.arccos(1/(1.33)))
            #total_ang = np.arctan(400/375)
            interval = 8
            #bound = (total_ang-max_ang)
            bound = 0.17453*interval

            lb = math.pi/2 - bound
            ub = math.pi/2 + bound

            c = ma.masked_where((self.angles[self.test_indices,0] > ub) | (self.angles[self.test_indices,0] < lb), self.test_indices)
            self.test_indices = c.compressed()
            #self.test_indices = self.test_indices[:32343]


            # For dataset with varying position:
            bound = find_bounds(self.positions[:,0,:], self.angles[:,:], self.labels[:], self.energies[:,0])

            c = ma.masked_where(bound[self.test_indices,2] < 200, self.test_indices)
            c = ma.masked_where(abs(self.positions[self.test_indices,0,1]) > 250, c)
            c = ma.masked_where((self.angles[self.test_indices,0] > bound[:,1]) | (self.angles[self.test_indices,0] < bound[:,0]), self.test_indices)

            bound = find_bounds(self.positions[self.test_indices,0,:], self.angles[self.test_indices,:], self.labels[self.test_indices], self.energies[self.test_indices,0])
            c = ma.masked_where((self.positions[self.test_indices,0,0]**2 + self.positions[self.test_indices,0,2]**2 + self.positions[self.test_indices,0,1]**2)**0.5 > 400, self.test_indices)
            c = ma.masked_where((self.angles[self.test_indices,0] > bound[:,1]) | (self.angles[self.test_indices,0] < bound[:,0]), c)
            #c = ma.masked_where(bound[self.test_indices,2] < 200, c)
            #c = ma.masked_where(bound[self.test_indices,2] > 400, c)
            self.test_indices = c.compressed()
            '''

            if self.event_data[i][0,:,:,:].shape[0] == 16:         
                self.a = None
                self.b = np.zeros((12, 40, 19), dtype=self.event_data[i].dtype)
                self.c = None
                d = np.array([[0,19],[0,20],
                                [1,17],[1,18],[1,19],[1,20],[1,21],[1,22],
                                [2,16],[2,17],[2,18],[2,19],[2,20],[2,21],[2,22], [2,23],
                                [3,15],[3,16],[3,17],[3,18],[3,19],[3,20],[3,21],[3,22],[3,23],[3,24],
                               [4,15],[4,16],[4,17],[4,18],[4,19],[4,20],[4,21],[4,22],[4,23],[4,24],
                               [5,14],[5,15],[5,16],[5,17],[5,18],[5,19],[5,20],[5,21],[5,22],[5,23],[5,24],[5,25],
                               [6,14],[6,15],[6,16],[6,17],[6,18],[6,19],[6,20],[6,21],[6,22],[6,23],[6,24],[6,25],
                               [7,15],[7,16],[7,17],[7,18],[7,19],[7,20],[7,21],[7,22],[7,23],[7,24],
                                [8,15],[8,16],[8,17],[8,18],[8,19],[8,20],[8,21],[8,22],[8,23],[8,24],
                               [9,16],[9,17],[9,18],[9,19],[9,20],[9,21],[9,22],[9,23],
                               [10,17],[10,18],[10,19],[10,20],[10,21],[10,22],
                               [11,19],[11,20]])
                self.d = np.concatenate((d,d), axis=0)
                self.d[96:,0] = self.d[96:,0] + 28
                self.e = None
                self.f = None
                self.g = None
        
        self.b = np.zeros((40, 40, 19), dtype=self.event_data[0].dtype);
        
        self.endcap_mPMT_order = np.array([[0,6],[1,7],[2,8],[3,9],[4,10],[5,11],[6,0],[7,1],[8,2],[9,3],[10,4],[11,5],[12,15],[13,16],[14,17],[15,12],[16,13],[17,14],[18,18]])
        self.datasets = np.array(np.arange(num_datasets))

            
    def __getitem__(self, index):
        '''
        self.a = self.event_data[self.datasets[0]][index,:,:,:19]
        #self.c = self.a[:,:,self.endcap_mPMT_order[:,1]]
        #self.c[12:28,:,:] = self.a[12:28,:,:19]
        self.c = self.a
        return np.squeeze(self.chrg_func(np.expand_dims(np.ascontiguousarray(np.transpose(self.c,[2,0,1])), axis=0), self.chrg_acc, apply=True)), self.labels[self.datasets[0]][index], self.energies[self.datasets[0]][index], self.angles[self.datasets[0]][index], index, self.positions[self.datasets[0]][index]
        '''
        np.random.shuffle(self.datasets)
        for i in np.arange(len(self.datasets)):
            
            if index < (self.labels[self.datasets[i]].shape[0]):
                label = self.label_map(self.labels[self.datasets[i]][index]) 
                if self.event_data[self.datasets[i]][index, :, :, :19].shape[0] == 16:

                    self.a = self.event_data[self.datasets[i]][index, :, :, :19]
                    self.c = np.concatenate((self.b,self.a,self.b), axis=0)
                    self.e = np.random.rand(192,19,2)
                    prob = random.randrange(1, 7, 1)/100
                    self.f = self.e[:,:,0] > prob
                    self.g = np.where(self.f, 0, self.e[:,:,1])
                    self.c[self.d[:,0], self.d[:,1]] = self.g

                    return np.squeeze(self.chrg_func(np.expand_dims(np.ascontiguousarray(np.transpose(self.c,[2,0,1])),axis=0), self.chrg_acc, apply=True)), label, self.energies[self.datasets[i]][index], self.angles[self.datasets[i]][index], index, self.eventids[self.datasets[i]][index], self.rootfiles[self.datasets[i]][index]

                else:
                    data = self.event_data[self.datasets[i]][index,:,:,:19]
                    #self.c = self.a[:,:,self.endcap_mPMT_order[:,1]]
                    #self.c[12:28,:,:] = self.a[12:28,:,:19]
                    return np.squeeze(self.chrg_func(np.expand_dims(np.ascontiguousarray(np.transpose(data,[2,0,1])), axis=0), self.chrg_acc, apply=True)), label, self.energies[self.datasets[i]][index], self.angles[self.datasets[i]][index], index, self.eventids[self.datasets[i]][index], self.rootfiles[self.datasets[i]][index]
        
        assert False, "empty batch"
        raise RuntimeError("empty batch")
        
        
    def __len__(self):
        if self.reduced_size is None:
            return self.labels[0].shape[0]
        else:
            return self.reduced_size


