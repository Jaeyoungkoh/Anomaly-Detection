TOTAL_CONFIG = {
        # DATASET CONFIG
        'SMD': {
            'train_dir': 'data/SMD/train/',
            'test_dir': 'data/SMD/test/',
            'test_label_dir': 'data/SMD/labels/',
            'interpretation_label_dir': 'data/SMD/interpretation_label/',
            'sub_data_name' : 'machine-1-1',   
            'slide_size' : 1,
        },
        'SWaT': {
            'train_path': 'data/SWaT/SWaT_Dataset_Normal_v1.pkl',
            'test_path': 'data/SWaT/SWaT_Dataset_Attack_v0.pkl',
            'sub_data_name' : None,
            'slide_size' : 1,         
        },
        'SMAP_MSL': {
            'train_dir': 'data/SMAP_MSL/train',
            'test_dir': 'data/SMAP_MSL/test',
            'test_label_path' : 'data/SMAP_MSL/labeled_anomalies.csv',
            'sub_data_name' : 'T-1',
            'slide_size' : 1,
        },  
        'synthetic': {
            'train_dir': 'data/synthetic/synthetic_data_with_anomaly-s-1.csv',
            'test_label_dir' : 'data/synthetic/test_anomaly.csv',
            'sub_data_name' : None,
            'slide_size' : 1,
        },  
        'WADI': {
            'train_path': 'data/WADI/WADI_14days.csv',
            'test_path': 'data/WADI/WADI_attackdata.csv',
            'test_label_path' : 'data/WADI/WADI_attacklabels.csv',
            'sub_data_name' : None,
            'slide_size' : 1,
        },               
        'COLLECTOR': {
            'train_path': 'data/COLLECTOR/train_dmqa8.csv',
            'test_path': 'data/COLLECTOR/test_dmqa8.csv',
            'test_label_path': 'data/COLLECTOR/test_label_dmqa8.csv',
            'sub_data_name' : None,
            'slide_size' : 1
        },
        # MODEL CONFIG
        'VTTSAT': {
            'win_size': 100,            
            'optimizer' : 'adam',
            'scaler' : 'standard',
            'lr': 0.0001,
            'model_type' : 'reconstruction'
        },
        'VTTPAT': {
            'win_size': 100,            
            'optimizer' : 'adam',
            'scaler' : 'standard',
            'lr': 0.0001,
            'model_type' : 'reconstruction'
        },
        'AnomalyTransformer': {
            'win_size': 100,            
            'optimizer' : 'adam',
            'scaler' : 'standard',
            'lr': 0.0001,
            'model_type' : 'reconstruction',
            'num_epochs' : 10            
        },        
        'TranAD': {
            'win_size': 10,            
            'optimizer' : 'adamw',
            'scaler' : 'minmax',
            'lr': 0.008, # 초기값
            'model_type' : 'reconstruction',
            'num_epochs' : 5            
        },
        'MTAD_GAT': {
            'win_size': 100,            
            'optimizer' : 'adam',
            'scaler' : 'minmax',
            'lr': 0.001,
            'model_type' : 'mix',
            'num_epochs' : 50            
        },        
        'GDN': {
            'win_size': 5,            
            'optimizer' : 'adam',
            'scaler' : 'minmax',
            'lr': 0.001,
            'model_type' : 'forecasting',
            'num_epochs' : 50         
        },
        'DualTransformer': {
            'win_size': 50,            
            'optimizer' : 'adamw',
            'lr': 0.0002,            
            'scaler' : 'minmax',
            'weight_decay' : 0.01,
            'model_type' : 'reconstruction'
        },
        'Proposed': {
            'win_size': 50,            
            'optimizer' : 'adamw',
            'lr': 0.0002,            
            'scaler' : 'minmax',
            'weight_decay' : 0.01,
            'model_type' : 'reconstruction'
        },
        'Proposed_v2': {
            'win_size': 200,            
            'optimizer' : 'adamw',
            'lr': 0.0002,            
            'scaler' : 'minmax',
            'weight_decay' : 0.01,
            'model_type' : 'reconstruction',
            'num_epochs' : 50
        },
        'Proposed_v3': {
            'win_size': 200,            
            'optimizer' : 'adamw',
            'lr': 0.0002,            
            'scaler' : 'minmax',
            'weight_decay' : 0.01,
            'model_type' : 'reconstruction'
        }
        }