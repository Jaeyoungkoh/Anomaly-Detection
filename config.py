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
            'win_size': 100,            
            'optimizer' : 'adam',
            'scaler' : 'standard',
            'lr': 0.0001, # 초기값
            'model_type' : 'reconstruction',
            'num_epochs' : 3,
            'slide_size' : 100            
        },    
        'TranAD': {
            'win_size': 10,            
            'optimizer' : 'adamw',
            'scaler' : 'minmax',
            'lr': 0.001, # 초기값
            'model_type' : 'reconstruction',
            'num_epochs' : 5,
            'slide_size' : 1            
        },
        'MTAD_GAT': {
            'win_size': 100,            
            'optimizer' : 'adam',
            'scaler' : 'minmax',
            'lr': 0.001,
            'model_type' : 'mix',
            'num_epochs' : 30,
            'slide_size' : 1            
        },        
        'GDN': {
            'win_size': 5,            
            'optimizer' : 'adam',
            'scaler' : 'minmax',
            'lr': 0.001,
            'model_type' : 'forecasting',
            'num_epochs' : 30,
            'slide_size' : 1         
        },        
        'DCdetector': {
            'win_size':105,            
            'optimizer' : 'adam',
            'scaler' : 'standard',
            'lr': 0.0001,
            'model_type' : 'reconstruction',
            'num_epochs' : 3,
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
            'win_size':100, 
            'optimizer' : 'adam',
            'scaler' : 'standard',
            'lr': 0.0001,
            'model_type' : 'reconstruction',
            'num_epochs' : 10, # COLLECTOR : 10, SWaT : 10, SMAP :3, MSL : 1, SMD : 10
            'slide_size' : 1        
        },      
        'USAD': {
            'win_size':5, # COLLECTOR : 5, SWaT : 12, SMAP :5, MSL : 5, SMD : 5
            'optimizer' : 'adam',
            'scaler' : 'minmax',
            'lr': 0.001,
            'model_type' : 'reconstruction',
            'num_epochs' : 250, # COLLECTOR : 250, SWaT : 70, SMAP :250, MSL : 250, SMD : 250
            'slide_size' : 1        
        },
        'DualTransformer': {
            'win_size': 50,            
            'optimizer' : 'adamw',
            'lr': 0.0002,            
            'scaler' : 'minmax',
            'weight_decay' : 0.01,
            'model_type' : 'reconstruction',
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
        'Proposed_v1': {
            'win_size': 200,            
            'optimizer' : 'adamw',
            'lr': 0.0002,            
            'scaler' : 'minmax',
            'weight_decay' : 0.01,
            'model_type' : 'reconstruction',
            'num_epochs' : 50,
            'slide_size' : 1
        },
        'Proposed_v2': {
            'win_size': 50,            
            'optimizer' : 'adamw',
            'lr': 0.0002,            
            'scaler' : 'minmax',
            'weight_decay' : 0.01,
            'model_type' : 'reconstruction',
            'num_epochs' : 50,
            'slide_size' : 1
        },
        'Proposed_v3': {
            'win_size': 200,            
            'optimizer' : 'adamw',
            'lr': 0.0002,            
            'scaler' : 'minmax',
            'weight_decay' : 0.01,
            'model_type' : 'reconstruction',
            'num_epochs' : 50,
            'slide_size' : 1
        },
        'Proposed_v4': {
            'win_size': 200,            
            'optimizer' : 'adamw',
            'lr': 0.0002,            
            'scaler' : 'minmax',
            'weight_decay' : 0.01,
            'model_type' : 'reconstruction',
            'num_epochs' : 50,
            'slide_size' : 1            
        },
        'Proposed_v5': {
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
            'num_epochs' : 50,
            'slide_size' : 1          
        }
        }