import numpy as np
from sklearn.metrics import ndcg_score

def hit_att(ascore, labels, ps = [100, 150]):
	# 한계점: 1등으로 찾으나 10등으로 찾으나, 범위 안에만 들어오면 똑같이 1건으로 취급 (순위에 따른 가중치 없음)
	res = {}
	for p in ps:
		hit_score = []
		for i in range(ascore.shape[0]):
			a, l = ascore[i], labels[i]
			a, l = np.argsort(a).tolist()[::-1], set(np.where(l == 1)[0]) # 내림차순 정렬(점수 높은 순서대로) & Label이 위치한 인덱스 추출
			if l:
				size = round(p * len(l) / 100) # i시점의 실제 이상치 개수 = l개일 때, 상위 탐색 범위 설정 
				a_p = set(a[:size])
				intersect = a_p.intersection(l) # 모델이 예측한 상위변수와 실제 상위변수(label) 사이의 교집합
				hit = len(intersect) / len(l) # (상위 탐색 범위 내 실제 이상치 개수) / (전체 실제 이상치 개수)
				hit_score.append(hit)
		res[f'Hit@{p}%'] = np.mean(hit_score)
	return res

def ndcg(ascore, labels, ps = [100, 150]):
	# 원리: 
	# 정답이 1등에 있을 때 가장 높은 점수를 주고, 2등, 3등, 4등으로 뒤로 밀릴수록 로그 함수를 이용해 점수를 Discount. 
	# 그 후 이상적인 최상의 정답 순서와 비교하여 0~1 사이의 값으로 정규화(Normalize)
	
	res = {}
	for p in ps:
		ndcg_scores = []
		for i in range(ascore.shape[0]):
			a, l = ascore[i], labels[i]
			labs = list(np.where(l == 1)[0])
			if labs:
				k_p = round(p * len(labs) / 100) # 실제 이상치 개수에 비례하여 탐색 범위(상위 K개)를 설정
				try:
					hit = ndcg_score(l.reshape(1, -1), a.reshape(1, -1), k = k_p)
				except Exception as e:
					return {}
				ndcg_scores.append(hit)
		res[f'NDCG@{p}%'] = np.mean(ndcg_scores)
	return res



