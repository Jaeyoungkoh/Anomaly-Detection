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
# seed=(316 423 777)
seed=(1 7 77 316 423 777 1004 1011 1234 2025)
# n_heads_temp=(1 4)
model_name=(Proposed_v2)
# model_name=(P)
# sub_data_name=(machine-2-9)
sub_data_name=(None)
mode=(train test)

echo "실험을 시작합니다..."

# 모든 조합 실행
for v1 in "${dataset[@]}"
do
  for v2 in "${seed[@]}"
  do
    for v3 in "${model_name[@]}"
    do
      for v4 in "${sub_data_name[@]}"
      do
        for v5 in "${mode[@]}"
        do        
          echo "-------------------------------------"
          echo "Running with dataset=${v1} seed=${v2} model_name=${v3} & sub_data_name=${v4} & mode=${v5}"
          # main.py 실행
          python main.py --dataset ${v1} --seed ${v2} --model_name ${v3} --sub_data_name ${v4} --mode ${v5}
        done
      done
    done  
  done 
done

echo "-------------------------------------"
echo "모든 실험이 완료되었습니다."
