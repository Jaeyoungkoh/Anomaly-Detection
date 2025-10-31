import torch
from data_factory.dataset import BuildDataset, load_dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

    # build dataset
    trn_dataset = BuildDataset(trn, window_size, slide_size, attacks=None, model_type=model_type)
    val_dataset = BuildDataset(val, window_size, slide_size, attacks=None, model_type=model_type)
    tst_dataset = BuildDataset(tst, window_size, window_size, attacks=label, model_type=model_type)

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
