#!/bin/bash

# 실행할 SEED 값들을 배열로 정의합니다.
SEED=(0 1 77 777 1004 1011 1234 2025)

# 배열의 각 SEED 값에 대해 반복 실행합니다.
for SEED in "${SEEDS[@]}"
do
  # 현재 어떤 SEED 값으로 실행하는지 터미널에 표시합니다.
  echo ">>> Running main_copy.py with seed: $SEED"

  # main_copy.py 스크립트를 --seed 인자와 함께 실행합니다.
  python main_copy.py --seed $SEED
done

# 모든 작업이 끝났음을 알립니다.
echo ">>> All runs are complete."
