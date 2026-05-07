import numpy as np
from pyod.models.iforest import IForest

class IForest_Wrapper:
    def __init__(self, args):
        # PyOD IForest 초기화
        self.model = IForest(
            n_estimators = args.n_estimators,
            contamination = args.contamination,
            n_jobs=-1,
            random_state=args.seed
        )
        self.model_name = 'IForest'

    def fit(self, train_data):
        # 3D (B, L, C) -> 2D (B, L*C) 변환
        batch_size, win_size, num_features = train_data.shape
        train_data_2d = train_data.reshape(batch_size, -1)
        
        self.model.fit(train_data_2d)

    def decision_function(self, test_data):
        # 3D (B, L, C) -> 2D (B, L*C) 변환
        batch_size, win_size, num_features = test_data.shape
        test_data_2d = test_data.reshape(batch_size, -1)
        
        # 점수 산출 (높을수록 이상치)[cite: 9]
        scores = self.model.decision_function(test_data_2d)
        return scores