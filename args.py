import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--seed', type=int, help='set seed if reproducibility is required', default=1) # seed 설정
    parser.add_argument("--dataset", type=str, default="SMD", choices = ['SWaT, SMD, SMAP_MSL, COLLECTOR'])
    parser.add_argument('--sub_data_name', type=str, default=None, help='dataset name')
    parser.add_argument('--model_name', type=str, default='TranAD', choices=['AnomalyTransformer', 
                                                                            'MTAD_GAT', 
                                                                            'TranAD',
                                                                            'VTTPAT',
                                                                            'VTTSAT',
                                                                            'Proposed'])
    parser.add_argument('--model_type', type=str, default=None, choices=['reconstruction', 'forecasting'])    
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--model_id", type=str, default=None, help="ID (datetime) of pretrained model to use, '-1' for latest, '-2' for second latest, etc")

    # Train
    parser.add_argument('--patience', type=int, default=10, help = 'Early Stopping')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--win_size', type=int, default=None)
    parser.add_argument('--valid_split_rate', type=float, default=0.8)

    # parser.add_argument('--anormly_ratio', type=float, default=1.00)

    # AnomalyTransformer   
    parser.add_argument('--k', type=int, default=3, help='anomaly Tranformer loss weight')     
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2, help='AnomalyAttention_dropout')
    parser.add_argument('--n_heads', type=int, default=8)    
    parser.add_argument('--e_layers', type=int, default=3)   
    parser.add_argument('--temperature', type=int, default=0.1)    
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
    parser.add_argument("--adjust_lr", type=str2bool, default=True) 

    # VTT
    parser.add_argument('--hidden_size', type=int, default=128, help='')    
    parser.add_argument('--n_layers_vtt', type=int, default=3, help='')
    parser.add_argument('--n_heads_vtt', type=int, default=4, help='')
    parser.add_argument('--resid_pdrop', type=float, default=0.1, help='')
    parser.add_argument('--attn_pdrop', type=float, default=0.1, help='')
    parser.add_argument('--time_emb', type=int, default=4, help='')
    parser.add_argument("--lradj", type=str, default='type1', help='')

    # GAT
    parser.add_argument('--kernel_size', type=int, default=7, help='1Dconv')     
    parser.add_argument('--feat_gat_embed_dim', type=int, default=None, help='feat_gat_embed_dim')     
    parser.add_argument('--time_gat_embed_dim', type=int, default=None, help='time_gat_embed_dim')
    parser.add_argument("--use_gatv2", type=str2bool, default=True) 
    parser.add_argument('--gru_n_layers', type=int, default=1, help='GRU layer 개수')     
    parser.add_argument('--gru_hid_dim', type=int, default=300, help='GRU 임베딩 차원')     
    parser.add_argument('--forecast_n_layers', type=int, default=3)   
    parser.add_argument('--forecast_hid_dim', type=int, default=300)       
    parser.add_argument('--recon_n_layers', type=int, default=1)   
    parser.add_argument('--recon_hid_dim', type=int, default=300)   
    parser.add_argument('--dropout_gat', type=float, default=0.2, help='MTAD-GAT_dropout')
    parser.add_argument('--alpha', type=float, default=0.2, help='')
    parser.add_argument("--scale_scores", type=str2bool, default=True, help='Anomaly score IQR 사용 여부') 
   
    # Model-agnostic Norm/Denorm
    parser.add_argument("--norm_type", type=str, default='revin', help=['revin', 'dish-ts', None])
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')  
    parser.add_argument('--subtract_last', type=int, default=1, help='0: subtract mean; 1: subtract last')  

    # ETC
    parser.add_argument("--save_auc_curve", type=str2bool, default=True) 
    parser.add_argument("--output_attention", type=str2bool, default=True) 
    parser.add_argument("--log_tensorboard", type=str2bool, default=True)    

    return parser