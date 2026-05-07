TOTAL_CONFIG = {
        # DATASET CONFIG
        'SMD': {
            'train_dir': 'data/SMD/train/',
            'test_dir': 'data/SMD/test/',
            'test_label_dir': 'data/SMD/labels/',
            'interpretation_label_dir': 'data/SMD/interpretation_label/',
            'sub_data_name' : 'machine-1-7'
        },
        'SWaT': {
            'train_path': 'data/SWaT/SWaT_Dataset_Normal_v1.pkl',
            'test_path': 'data/SWaT/SWaT_Dataset_Attack_v0.pkl',
            'sub_data_name' : None     
        },
        'SMAP_MSL': {
            'train_dir': 'data/SMAP_MSL/train',
            'test_dir': 'data/SMAP_MSL/test',
            'test_label_path' : 'data/SMAP_MSL/labeled_anomalies.csv',
            'sub_data_name' : 'T-1'
        },  
        'synthetic': {
            'train_dir': 'data/synthetic/synthetic_data_with_anomaly-s-1.csv',
            'test_label_dir' : 'data/synthetic/test_anomaly.csv',
            'sub_data_name' : None
        },  
        'synthetic_cp': {
            'train_path': 'data/synthetic_cp/train_data.csv',
            'test_path': 'data/synthetic_cp/test_data.csv',
            'test_label_path' : 'data/synthetic_cp/test_label.csv',
            'sub_data_name' : None    
        },  
        'WADI': {
            'train_path': 'data/WADI/WADI_14days.csv',
            'test_path': 'data/WADI/WADI_attackdata.csv',
            'test_label_path' : 'data/WADI/WADI_attacklabels.csv',
            'sub_data_name' : None
        },               
        'COLLECTOR': {
            'train_path': 'data/COLLECTOR/train_dmqa8.csv',
            'test_path': 'data/COLLECTOR/test_dmqa8.csv',
            'test_label_path': 'data/COLLECTOR/test_label_dmqa8.csv',
            'sub_data_name' : None
        },
        # MODEL CONFIG
        'VTTSAT': {
            'win_size': 100,            
            'optimizer' : 'adam',
            'scaler' : 'standard',
            'lr': 0.0001,
            'model_type' : 'reconstruction',
            'slide_size' : 1
        },
        'VTTPAT': {
            'win_size': 100,            
            'optimizer' : 'adam',
            'scaler' : 'standard',
            'lr': 0.0001,
            'model_type' : 'reconstruction',
            'slide_size' : 1
        },    
        'AnomalyTransformer': {
            'win_size': 100,                            # ALL : 100
            'optimizer' : 'adam',
            'scaler' : 'standard',               
            'lr': 0.0001,
            'model_type' : 'reconstruction',
            'batch_size' : 256,                         # ALL : 256
            'num_epochs' : 3,                           # COLLECTOR : 3, SWaT : 3, SMAP :3, MSL : 3, SMD : 10
            'slide_size' : 100            
        },    
        'TranAD': {
            'win_size': 10,                             # ALL : 10
            'optimizer' : 'adamw',
            'scaler' : 'minmax',                        # ALL : Min-max
            'lr': 0.001,                                # COLLECTOR : 0.001, SWaT : 0.008, SMAP :0.001, MSL : 0.002, SMD : 0.0001
            'model_type' : 'reconstruction',
            'batch_size' : 128,                         # ALL : 128
            'num_epochs' : 5,                           # ALL : 5
            'slide_size' : 1            
        },
        'MTAD_GAT': {
            'win_size': 100,                            # ALL : 100 (paper)
            'optimizer' : 'adam',
            'scaler' : 'minmax',
            'lr': 0.001,
            'model_type' : 'mix',
            'batch_size' : 256,                         # github            
            'num_epochs' : 100,                         # paper
            'slide_size' : 1            
        },        
        'GDN': {
            'win_size': 5,                              # ALL : 5
            'optimizer' : 'adam',
            'scaler' : 'minmax',
            'lr': 0.001,
            'model_type' : 'forecasting',
            'batch_size' : 128,                         # github                      
            'num_epochs' : 50,                          # paper (early stopping 10)
            'slide_size' : 1         
        },        
        'DCdetector': {
            'win_size':105,                             # COLLECTOR : 105, SWaT : 105, SMAP :105, MSL : 90, SMD : 105          
            'optimizer' : 'adam',
            'scaler' : 'standard',
            'lr': 0.0001,
            'model_type' : 'reconstruction',
            'patch_size' : [3, 5, 7],                   # COLLECTOR : [3, 5, 7], SWaT : [3, 5, 7], SMAP :[3, 5, 7], MSL : [3, 5], SMD : [5, 7]    
            'batch_size' : 128,                         # papar                    
            'num_epochs' : 3,                           # papar
            'slide_size' : 1        
        },        
        'LSTM_AE': {
            'win_size':30,            
            'optimizer' : 'adam',
            'scaler' : 'minmax',
            'lr': 0.001,
            'model_type' : 'reconstruction',
            'num_epochs' : 30,
            'slide_size' : 1        
        },      
        'OmniAnomaly': {
            'win_size':100,            
            'optimizer' : 'adam',
            'scaler' : 'minmax',
            'lr': 0.002,
            'model_type' : 'reconstruction',
            'num_epochs' : 30,
            'slide_size' : 1        
        },      
        'TimesNet': {
            'win_size':100,                             # ALL
            'optimizer' : 'adam',
            'scaler' : 'standard',
            'lr': 0.0001,
            'model_type' : 'reconstruction',
            'batch_size' : 128,                         # papar                
            'num_epochs' : 10,                          # COLLECTOR : 10, SWaT : 10, SMAP :3, MSL : 1, SMD : 10
            'slide_size' : 1        
        },      
        'USAD': {
            'win_size':12,                              # COLLECTOR : 5, SWaT : 12, SMAP :5, MSL : 5, SMD : 5
            'optimizer' : 'adam',
            'scaler' : 'minmax',
            'lr': 0.001,
            'model_type' : 'reconstruction',
            'batch_size' : 1024,                         # 미정             
            'hidden_dim_usad' : 15,                     # COLLECTOR : 15, SWaT : 40, SMAP : 55, MSL : 33, SMD : 38
            'num_epochs' : 250,                         # COLLECTOR : 250, SWaT : 70, SMAP :250, MSL : 250, SMD : 250
            'slide_size' : 1            
        },     
        'LSTM_VAE': {
            'win_size':50,                              # paper (https://github.com/TimyadNyda/Variational-Lstm-Autoencoder/tree/master)
            'optimizer' : 'adam',
            'scaler' : 'minmax',
            'lr': 0.001,
            'model_type' : 'reconstruction',
            'batch_size' : 128,                         # 미정             
            'num_epochs' : 200,
            'slide_size' : 1        
        },     
        'IForest': {
            'win_size':1,                   
            'optimizer' : 'adam',
            'scaler' : 'minmax',
            'lr': 0.001,
            'model_type' : 'reconstruction',
            'batch_size' : 128,              
            'num_epochs' : 1,
            'slide_size' : 1        
        },
        'Proposed': {
            'win_size': 200,            
            'optimizer' : 'adamw',
            'lr': 0.0002,            
            'scaler' : 'minmax',
            'weight_decay' : 0.01,
            'model_type' : 'reconstruction',
            'num_epochs' : 50,
            'slide_size' : 1
        },
        'Proposed_v6': {
            'win_size': 200,            
            'optimizer' : 'adamw',
            'lr': 0.0002,            
            'scaler' : 'minmax',
            'weight_decay' : 0.01,
            'model_type' : 'reconstruction',
            'batch_size' : 32,                         
            'num_epochs' : 50,
            'slide_size' : 1          
        }
        }