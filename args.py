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
    parser.add_argument("--dataset", type=str, default='WADI', choices = ['SWaT', 'SMD', 'SMAP_MSL', 'COLLECTOR', 'MSDS', 'synthetic', 'WADI'])
    parser.add_argument('--sub_data_name', type=str, default=None, help='dataset name')
    parser.add_argument('--model_name', type=str, default='Proposed_v2', choices=['AnomalyTransformer', 
                                                                            'MTAD_GAT',
                                                                            'GDN', 
                                                                            'TranAD',
                                                                            'VTTPAT',
                                                                            'VTTSAT',
                                                                            'DualTransformer',
                                                                            'Proposed',
                                                                            'Proposed_v2'])
    parser.add_argument('--model_type', type=str, default=None, choices=['reconstruction', 'forecasting', 'mix'])    
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--model_id", type=str, default=None, help="ID (datetime) of pretrained model to use, '-1' for latest, '-2' for second latest, etc")

    # Train
    parser.add_argument('--patience', type=int, default=10, help = 'Early Stopping')
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--win_size', type=int, default=None)
    parser.add_argument('--valid_split_rate', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type = float, default=0.01)

    # Proposed_v2
    parser.add_argument('--d_model_temp', type=int, default=512)   # 512  
    parser.add_argument('--d_ff_temp', type=int, default=None, choices = [512, None], help='if None, d_ff_temp = 4*d_model_temp')
    parser.add_argument('--n_heads_temp', type=int, default=4)    
    parser.add_argument('--e_layers_temp', type=int, default=4)
    parser.add_argument('--dropout_temp', type=float, default=0.2, help='dropout')

    parser.add_argument('--d_model_gat', type=int, default=None, help='d_model_gat')  # 200
    parser.add_argument('--d_ff_channel', type=int, default=None, choices = ['win_size', None], help='if None, d_ff = 4*d_model_gat')
    parser.add_argument('--n_heads_gat', type=int, default=1, help='num_head_gat')     
    parser.add_argument('--e_layers_gat', type=int, default=1)        
    parser.add_argument('--dropout_gat', type=float, default=0.2, help='MTAD-GAT_dropout')
    
    parser.add_argument('--fore_n_layers', type=int, default=2)   # 4
    parser.add_argument('--fore_hid_dim', type=int, default=256)  # 512
    parser.add_argument('--dropout_fore', type=float, default=0.2, help='MTAD-GAT_dropout')
    parser.add_argument("--gat_type", type=str, default='gatv2_mtadgat', choices = ['gat', 'gatv2_constr', 'gatv2_full', 'gatv2_mtadgat'], help='GAT version') 
    parser.add_argument("--use_node_embedding", type=str2bool, default=False)
    parser.add_argument("--use_gatv2", type=str2bool, default=True)
    parser.add_argument("--concat", type=str2bool, default=True)

    parser.add_argument("--share_weights", type=str2bool, default=False)
    parser.add_argument("--add_self_loops", type=str2bool, default=True)
    parser.add_argument("--bias", type=str2bool, default=False)
    parser.add_argument("--use_residual", type=str2bool, default=False)
    parser.add_argument("--use_layer_norm", type=str2bool, default=False)
    parser.add_argument("--use_activation", type=str2bool, default=False)            

    # AnomalyTransformer   
    parser.add_argument('--k', type=int, default=3, help='anomaly Tranformer loss weight')     
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512, choices = [512, None], help='if None, d_ff = 4*d_model')
    parser.add_argument('--dropout', type=float, default=0.0, help='AnomalyAttention_dropout')
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=3)   
    parser.add_argument('--temperature', type=int, default=50)    
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

    # MTAD-GAT
    parser.add_argument('--kernel_size', type=int, default=7, help='1Dconv')     
    parser.add_argument('--feat_gat_embed_dim', type=int, default=None, help='feat_gat_embed_dim')     
    parser.add_argument('--time_gat_embed_dim', type=int, default=None, help='time_gat_embed_dim')    
    parser.add_argument('--gru_n_layers', type=int, default=1, help='GRU layer 개수')     
    parser.add_argument('--gru_hid_dim', type=int, default=300, help='GRU 임베딩 차원')     
    parser.add_argument('--forecast_n_layers', type=int, default=3)
    parser.add_argument('--forecast_hid_dim', type=int, default=300)       
    parser.add_argument('--recon_n_layers', type=int, default=1)   
    parser.add_argument('--recon_hid_dim', type=int, default=300)   
    parser.add_argument('--alpha', type=float, default=0.2, help='')
    parser.add_argument('--gamma', type=float, default=1, help='')
    parser.add_argument("--scale_scores", type=str2bool, default=False, help='Anomaly score IQR 사용 여부') 
    parser.add_argument('--dropout_mtadgat', type=float, default=0.3, help='AnomalyAttention_dropout')

    # GDN
    parser.add_argument('--embed_dim', help='embedding dimension', type = int, default=64) # 64
    parser.add_argument('--save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('--out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('--out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=64) # 256
    parser.add_argument('--val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('--topk', help='topk num', type = int, default=15)
    parser.add_argument("--save_attention", type=str2bool, default=True)  

    # Model-agnostic Norm/Denorm
    parser.add_argument("--norm_type", type=str, default='revin', help=['revin', 'dish-ts', None])
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')  
    parser.add_argument('--subtract_last', type=int, default=1, help='0: subtract mean; 1: subtract last')  

    # ETC
    parser.add_argument("--save_auc_curve", type=str2bool, default=True) 
    parser.add_argument("--output_attention", type=str2bool, default=True) 
    parser.add_argument("--log_tensorboard", type=str2bool, default=True)    

    return parser