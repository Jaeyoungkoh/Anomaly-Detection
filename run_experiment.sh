'''
[실행 방법]
1. 스크립트에 실행 권한 부여
chmod +x run_experiment.sh
ch
2. 스크립트 실행
./run_experiment.sh
'''

#!/bin/bash

# 실행할 파라미터 배열 정의
model_name=('Proposed' 'MTAD_GAT' 'GDN' 'AnomalyTransformer' 'TranAD' 'VTTPAT' 'VTTSAT')

mode=('train' 'test')

echo "실험을 시작합니다..."

# 모든 조합 실행
for model in "${model_name[@]}"
do
  for md in "${mode[@]}"
  do
    echo "-------------------------------------"
    echo "Running with model=${model} and mode=${md}"
    
    # main.py 실행
    python main.py --model_name ${model} --mode ${md}
    
  done
done

echo "-------------------------------------"
echo "모든 실험이 완료되었습니다."
