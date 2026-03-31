'''
[실행 방법]
1. 스크립트에 실행 권한 부여
chmod +x run_experiment.sh

2. 스크립트 실행
./run_experiment.sh
'''

#!/bin/bash

#실행할 파라미터 배열 정의
dataset=(SMD)
# n_heads_temp=(1 4)
# seed=(1 7 77 316 423 777 1004 1011 1234 2025)
seed=(2025)
# n_heads_temp=(1 4)
model_name=(Proposed)
# use_decomp=(True)
# use_denorm=(False)
n_heads_temp=(1)
use_temporal=(True)
model_id=(30032026_131624)
sub_data_name=(machine-3-10)
# sub_data_name=(C-1 D-14 D-15 D-16 F-8 M-1 M-2)
# sub_data_name=(A-7 D-7 E-3 F-2 G-7 P-7 S-1)
# sub_data_name=(A-7 D-7 E-3 F-2 G-7 P-7 S-1 C-1 D-14 D-15 D-16 F-8 M-1 M-2)
# sub_data_name=(machine-1-1 machine-1-6 machine-1-7 machine-2-9 machine-3-4 machine-3-10)
mode=(test)

echo "실험을 시작합니다..."

# 모든 조합 실행
for v1 in "${dataset[@]}"
do
  for v2 in "${seed[@]}"
  do
    for v3 in "${model_name[@]}"
    do
      for v4 in "${n_heads_temp[@]}"
      do
        for v5 in "${model_id[@]}"
        do    
          for v6 in "${sub_data_name[@]}"
          do
            for v7 in "${mode[@]}"
            do        
              echo "-------------------------------------"
              echo "Running with dataset=${v1} seed=${v2} model_name=${v3} & n_heads_temp=${v4} & model_id=${v5} & sub_data_name=${v6} & mode=${v7}"
              # main.py 실행
              python main.py --dataset ${v1} --seed ${v2} --model_name ${v3} --n_heads_temp ${v4} --model_id ${v5} --sub_data_name ${v6} --mode ${v7}
            done
          done
        done
      done
    done  
  done 
done

echo "-------------------------------------"
echo "모든 실험이 완료되었습니다."
