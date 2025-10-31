import os
from args import get_parser
from torch.backends import cudnn
from utils.utils import *
import datetime
import argparse
from solver import Solver
from config import TOTAL_CONFIG
import random
import json

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    # config와 args 병합 (우선순위 args>config)
    dataset_config = TOTAL_CONFIG[args.dataset]
    model_config = TOTAL_CONFIG[args.model_name]
    config_merged = {**dataset_config, **model_config} # ** : dict unpacking 연산자
    for key, value in config_merged.items():
            if getattr(args, key, None) is None: # args에 해당 값이 설정되지 않았을 경우에만
                setattr(args, key, value)        # config의 값을 args로 설정

    # 재현성을 위한 시드 설정
    random.seed(args.seed)                           # Python random seed
    np.random.seed(args.seed)                        # NumPy seed
    torch.manual_seed(args.seed)                     # PyTorch CPU seed
    torch.cuda.manual_seed(args.seed)                # PyTorch GPU seed (single-GPU)  
    torch.backends.cudnn.deterministic = True        # deterministic 연산 강제 (재현 가능성 ↑)
    torch.backends.cudnn.benchmark = True   

    args.output_path = f'output/{args.dataset}/{args.model_name}' # output/dataset/model_name
    args.log_dir = f'{args.output_path}/logs'

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    # TRAIN SECTION
    if args.mode == 'train' :

        id = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        args.save_path = f"{args.output_path}/{id}" # output/COLLECTOR/id(실행시간)

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        # Save config
        args_path = f"{args.save_path}/config.txt"
        with open(args_path, "w") as f:
            json.dump(args.__dict__, f, indent=2)

        args_summary = str(vars(args))
        print('------------ Options -------------')    
        for k, v in sorted(config_merged.items()):
            print(f'{k}: {v}')
        print('-------------- End ----------------')

        solver = Solver(args)
        solver.train()

    # TEST SECTION
    elif args.mode == 'test':

        # model_id 미선택일 경우, 가장 최근 훈련된 모델 가져오기
        if args.model_id is None:
            dir_path = f"output/{args.dataset}/{args.model_name}"
            dir_content = os.listdir(dir_path)
            subfolders = [subf for subf in dir_content if os.path.isdir(f"{dir_path}/{subf}") and subf != "logs"]
            date_times = [datetime.datetime.strptime(subf, '%d%m%Y_%H%M%S') for subf in subfolders]
            date_times.sort()
            model_datetime = date_times[-1]
            model_id = model_datetime.strftime('%d%m%Y_%H%M%S')

        # args에 파일 이름 직접 지정 '%d%m%Y_%H%M%S'
        else:
            model_id = args.model_id

        model_path = f"output/{args.dataset}/{args.model_name}/{model_id}"

        if not os.path.isfile(f'{model_path}/model.pt'):
            raise Exception(f"<{model_path}/model.pt does not exist")
        
        # Pre-trained 모델에서 사용된 설정값 그대로 가져오기
        print(f'Using model from {model_path}') 
        model_parser = argparse.ArgumentParser()
        model_args, unknown = model_parser.parse_known_args()
        model_args_path = f"{model_path}/config.txt"

        with open(model_args_path, 'r') as f:
            model_args.__dict__ = json.load(f) # json.load(f)로 읽은 설정 값을 model_args의 속성으로 모두 할당

        # Check that model is trained on specified dataset
        if args.dataset.lower() != model_args.dataset.lower():
            raise Exception(f"Model trained on {model_args.dataset}, but asked to predict {args.dataset}.")
        
        model_args.mode = 'test'

        solver = Solver(model_args)
        solver.test()