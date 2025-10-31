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
            'sub_data_name' : 'F-7',
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
            'model_type' : 'reconstruction'
        },        
        'TranAD': {
            'win_size': 10,            
            'optimizer' : 'adamw',
            'scaler' : 'minmax',
            'lr': 0.01, # 초기값
            'model_type' : 'reconstruction'
        },
        'MTAD-GAT': {
            'win_size': 100,            
            'optimizer' : 'adam',
            'scaler' : 'minmax',
            'lr': 0.001,
            'model_type' : 'reconstruction'
        },        
        'Proposed': {
            'win_size': 50,            
            'optimizer' : 'adam',
            'scaler' : 'minmax',
            'lr': 0.0001,
            'model_type' : 'reconstruction'
        }
        }