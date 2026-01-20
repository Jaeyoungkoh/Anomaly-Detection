import torch
from data_factory.dataset import BuildDataset, load_dataset, construct_data, SlidingWindowDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import itertools # <--- [추가] GDN의 edge_index 생성용

def get_dataloader(args):
    
    scaler = args.scaler
    assert scaler in (None, 'minmax', 'minmax_square', 'minmax_m1p1', 'standard')

    window_size = args.win_size
    batch_size = args.batch_size
    slide_size = args.slide_size
    model_type = args.model_type

    # load dataset (data, timestamp, label)
    trn, val, tst, label = load_dataset(args)

    # scaling (minmax, minmax square, minmax m1p1, standard)
    if scaler is not None:
        if scaler == 'minmax':
            scaler = MinMaxScaler()
            scaler.fit(trn)
            trn = scaler.transform(trn)
            val = scaler.transform(val)
            tst = scaler.transform(tst)
        elif scaler == 'minmax_square':
            scaler = MinMaxScaler()
            scaler.fit(trn)
            trn = scaler.transform(trn) ** 2
            val = scaler.transform(val) ** 2
            tst = scaler.transform(tst) ** 2
        elif scaler == 'minmax_m1p1':
            trn = 2 * (trn / trn.max(axis=0)) - 1
            val = 2 * (val / val.max(axis=0)) - 1
            tst = 2 * (tst / tst.max(axis=0)) - 1
        elif scaler == 'standard':
            scaler = StandardScaler()
            scaler.fit(trn)
            trn = scaler.transform(trn)
            val = scaler.transform(val)
            tst = scaler.transform(tst)
        print(f'{scaler} Normalization done')

    if args.model_name in ['GDN', 'Proposed_v2']:
    # 1. Node 개수 파악 및 Fully-Connected Edge Index 생성
        node_num = trn.shape[1] # (Time, Features)에서 Features 개수
        edges = list(itertools.permutations(range(node_num), 2))
        sources = [edge[0] for edge in edges]
        targets = [edge[1] for edge in edges] 
        edge_index_list = [targets, sources]
        fc_edge_index = torch.tensor(edge_index_list, dtype=torch.long)

        # 2. [GDN] construct_data로 포맷 변경 (스케일링된 NumPy 배열 사용)
        trn_data_gdn = construct_data(trn, labels=0)
        val_data_gdn = construct_data(val, labels=0)
        tst_data_gdn = construct_data(tst, labels=label)
        
        # 3. [GDN] SlidingWindowDataset 생성
        cfg = {'slide_window': window_size, 'slide_stride': slide_size, 'model_type' : model_type}
        trn_dataset = SlidingWindowDataset(trn_data_gdn, fc_edge_index, mode='train', config=cfg)
        val_dataset = SlidingWindowDataset(val_data_gdn, fc_edge_index, mode='train', config=cfg)
        tst_dataset = SlidingWindowDataset(tst_data_gdn, fc_edge_index, mode='test', config=cfg)

    else: 
        # build dataset
        trn_dataset = BuildDataset(trn, window_size, slide_size, attacks=None, model_type=model_type)
        val_dataset = BuildDataset(val, window_size, slide_size, attacks=None, model_type=model_type)
        if model_type == 'reconstruction':
            tst_dataset = BuildDataset(tst, window_size, window_size, attacks=label, model_type=model_type)
        elif model_type in ['forecasting', 'mix']:
            tst_dataset = BuildDataset(tst, window_size, slide_size, attacks=label, model_type=model_type)

    # torch dataloader
    trn_dataloader = torch.utils.data.DataLoader(trn_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                #  num_workers=8,
                                                #  pin_memory=True,                                                 
                                                 drop_last=False)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                #  num_workers=8,
                                                #  pin_memory=True,                                                 
                                                 drop_last=False)

    tst_dataloader = torch.utils.data.DataLoader(tst_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                #  num_workers=8,
                                                #  pin_memory=True,                                                 
                                                 drop_last=False)

    return trn_dataloader, val_dataloader, tst_dataloader
