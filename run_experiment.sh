'''
[실행 방법]
1. 스크립트에 실행 권한 부여
chmod +x run_experiment.sh

2. 스크립트 실행
./run_experiment.sh
'''

#!/bin/bash

#실행할 파라미터 배열 정의
dataset=(COLLECTOR)
# n_heads_temp=(1 4)
# seed=(423 777 1004 1011 1234 2025)
# n_heads_temp=(1 4)
# model_name=(TranAD GDN MTAD_GAT)
model_name=(Proposed_v6)
# use_decomp=(True)
# use_denorm=(False)
e_layers_gat=(2)
# affine=(1)
n_heads_gat=(1 2)
# model_id=(13042026_145558)
use_residual=(True False)
# sub_data_name=(machine-1-1 machine-1-6 machine-1-7 machine-2-9 machine-3-4 machine-3-10)
# sub_data_name=(C-1 D-14 D-15 D-16 F-8 M-1 M-2)
# sub_data_name=(A-7 D-7 E-3 F-2 G-7 P-7 S-1)
# sub_data_name=(A-7 D-7 E-3 F-2 G-7 P-7 S-1 C-1 D-14 D-15 D-16 F-8 M-1 M-2)
# seed=(1 7 13 17 77 101 123 212 316 423 777 999 1004 1011 1234 1337 2002 2025 3333 7777)
# seed=(1 7 13 17 77 101 123 212 316 423 777 999 1004 1011 1234 1337 2002 2025 3333 7777)
seed=(1 7 77 316 423 777 1004 1011 1234 3333)
# seed=(1337 2002 3333)
mode=(train test)
# mode=(test)

# 모든 조합 실행
for v1 in "${dataset[@]}"
do
  for v2 in "${model_name[@]}"
  do
    for v3 in "${e_layers_gat[@]}"
    do
      for v4 in "${n_heads_gat[@]}"
      do
        
        # # [수정된 부분] e_layers_gat와 n_heads_gat가 모두 1일 때 건너뛰기
        # if [ "$v3" -eq 1 ] && [ "$v4" -eq 1 ]; then
        #   echo "-------------------------------------"
        #   echo "Skipping... e_layers_gat=1 & n_heads_gat=1"
        #   continue
        # fi

        for v5 in "${use_residual[@]}"
        do    
          for v6 in "${seed[@]}"
          do
            for v7 in "${mode[@]}"
            do        
              echo "-------------------------------------"
              echo "Running with dataset=${v1} model_name=${v2} e_layers_gat=${v3} & n_heads_gat=${v4} & use_residual=${v5} & seed=${v6} & mode=${v7}"
              # main.py 실행
              python main.py --dataset ${v1} --model_name ${v2} --e_layers_gat ${v3} --n_heads_gat ${v4} --use_residual ${v5} --seed ${v6} --mode ${v7}
            done
          done
        done
      done
    done  
  done 
done

echo "-------------------------------------"
echo "모든 실험이 완료되었습니다."
