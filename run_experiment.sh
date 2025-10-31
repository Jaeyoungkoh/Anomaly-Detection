'''
[실행 방법]
1. 스크립트에 실행 권한 부여
chmod +x run_experiments.sh

2. 스크립트 실행
./run_experiments.sh
'''

#!/bin/bash

# 실행할 파라미터 배열 정의
SEED = (0 11 17 111)
LRS = (0.001 0.002 0.003)

echo "실험을 시작합니다..."

# 모든 조합 실행
for seed in "${SEED[@]}"
do
  for lr in "${LR[@]}"
  do
    echo "-------------------------------------"
    echo "Running with seed=${seed} and lr=${lr}"
    
    # main.py 실행
    python main.py --seed ${seed} --lr ${lr}
    
  done
done

echo "-------------------------------------"
echo "모든 실험이 완료되었습니다."
