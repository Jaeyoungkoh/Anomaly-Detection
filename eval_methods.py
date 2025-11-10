import numpy as np
import more_itertools as mit
from utils.spot import SPOT, dSPOT
from collections import Counter
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, precision_score, recall_score, average_precision_score
import torch
import math

def compute_reconstruction_probability_featurewise(recon_inputs, recon_means, recon_logvars):

    """
    Parameters:
    - x:             (b, n, k)  ground truth
    - recon_mean:    (b, n, k)  decoder predicted mean
    - recon_logvar:  (b, n, k)  decoder predicted log variance

    Returns:
    - p:             (b, k)     probability per feature at last timestep
    """

    # 마지막 timestep 사용
    input_last = recon_inputs[:, -1, :]  # (b, k)
    mean_last = recon_means[:, -1, :]  # (b, k)
    logvar_last = recon_logvars[:, -1, :]  # (b, k)

    sigma2 = torch.exp(logvar_last)       # (b, k)
    recon_error = (input_last - mean_last) ** 2  # (b, k)

    log_p = -0.5 * (torch.log(2 * torch.pi * sigma2) + recon_error / sigma2)
    p = torch.exp(log_p)  # (b, k)

    return p


def adjust_predicts(score, label, threshold, pred=None, calc_latency=False):
    """
    이상 구간 전체를 탐지로 확장 (구간 기반 평가 대응)
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
            score (np.ndarray): The anomaly score
            label (np.ndarray): The ground-truth label
            threshold (float): The threshold of anomaly score.
                    A point is labeled as "anomaly" if its score is lower than the threshold.
            pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
            calc_latency (bool):
    Returns:
            np.ndarray: predict labels

    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    
    # label이 없으면 평가 불가능 → threshold 기반으로 0/1 이진 예측만 반환
    if label is None:
        predict = score > threshold
        return predict, None
    
    # pred가 제공되지 않은 경우, 직접 계산
    if pred is None:
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        predict = score > threshold 
    # label 있고 pred 제공되는 경우    
    else:
        predict = pred 

    actual = label > 0.1 # float label의 경우를 대비하여, 0.1 이상이면 이상치로 간주 (boolean array)
    anomaly_state = False
    anomaly_count = 0
    latency = 0


    for i in range(len(predict)):

        '''
        현재 시점 i가 실제 이상 구간에 속해 있고 (actual[i] == True)
        모델이 이 시점에서 처음으로 이상을 탐지했고 (predict[i] == True, anomaly_state == False)
        → 이 경우, 해당 이상 구간을 탐지한 것으로 간주하고 구간 전체에 대해 보정 시작
        '''

        # 5-1. 구간 시작 → latency 계산 및 보정
        if any(actual[max(i, 0) : i + 1]) and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        
        # 5-2. 이상치가 끝났다고 판단하면 anomaly_state 종료                
        elif not actual[i]:
            anomaly_state = False

        # 5-3. 이상치 구간 중일 땐 계속 탐지 유지
        if anomaly_state:
            predict[i] = True
        
    # calc_latency=True면 평균 지연 시간도 함께 반환
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict, None

def adjust_predicts2(score, label, threshold, pred=None, calc_latency=False, window=200):
    """
    anomaly score 또는 pred를 기반으로 이상 탐지 지점을 찾아,
    해당 시점부터 과거 window 길이만큼 1로 보정하는 함수.

    Args:
        score (np.ndarray): anomaly score (pred가 None일 때 사용)
        label (np.ndarray): 사용되지 않음 (호환성 유지용)
        threshold (float): threshold 값
        pred (np.ndarray or None): 초기 예측 결과 (있으면 이것을 보정 대상)
        calc_latency (bool): 무시됨 (항상 None 반환)
        window (int): 보정 길이 (기본값 200)

    Returns:
        np.ndarray: 보정된 예측 결과 (binary vector)
        None: latency 값 placeholder
    """
    
    # 입력 유효성 확인
    if pred is None and score is None:
        raise ValueError("score와 pred 중 하나는 반드시 제공되어야 합니다.")

    if pred is not None:
        if not isinstance(pred, np.ndarray):
            pred = np.array(pred)
        # pred에서 1인 지점을 anomaly로 간주
        anomaly_indices = np.where(pred == 1)[0]
    else:
        # score에서 threshold를 넘는 지점을 anomaly로 간주
        anomaly_indices = np.where(score > threshold)[0]

    # 보정된 예측값 생성
    predict = np.zeros_like(pred if pred is not None else score, dtype=int)

    '''
    과거 윈도우 모두 1로 처리할 때 사용
    for i in anomaly_indices:
         start = max(0, i - window + 1)
         predict[start:i + 1] = 1   
    '''

    # 해당 시점만 이상으로 표시
    for i in anomaly_indices:
        predict[i] = 1

    return predict, None


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
            predict (np.ndarray): the predict label
            actual (np.ndarray): np.ndarray
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    predict = predict.reshape(-1,1)
    actual = actual.reshape(-1,1)
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN



def PA_percentile(score, label,
                  threshold=None,
                  pred=None,
                  K=100,
                  calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    anomalies = [] # anomaly 구간(리스트) 모음

    for i in range(len(actual)):
        if actual[i]:
            if not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                anomalies.append([i, i])
            else:
                anomalies[-1][-1] = i
        else:
            anomaly_state = False

    for i, [start, end] in enumerate(anomalies):
        collect = Counter(predict[start:end + 1].flatten().tolist())[1] # start~end까지 slice & 0, 1이 각각 몇번 나오는지 Count. [1]은 1이 몇번 나오는지
        anomaly_count += collect
        collect_ratio = collect / (end - start + 1)

        if collect_ratio * 100 >= K and collect > 0:
            predict[start:end + 1] = True
            latency += (end - start + 1) - collect

    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

# def calc_seq(score, label, threshold):
#     # predict, latency = adjust_predicts(score, label, threshold, calc_latency=False)
#     predict, latency = adjust_predicts2(score, label, threshold, calc_latency=False, window=200)
#     return calc_point2point(predict, label), latency

def calc_seq(score, label, threshold, K=0, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        roc_auc = roc_auc_score(label, score)
        auprc = average_precision_score(label, score)
        #predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        predict, latency = PA_percentile(score, label, threshold, K=K, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(roc_auc)
        t.append(auprc)
        t.append(predict)
        t.append(latency)
        return t
    else:
        roc_auc = roc_auc_score(label, score)
        auprc = average_precision_score(label, score)
        # predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        predict = PA_percentile(score, label, threshold, K=K, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(roc_auc)
        t.append(auprc)
        t.append(predict)
        return t


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, K=0, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """

    print(f"Finding best f1-score by searching for threshold..")

    if step_num is None or end is None:
        end = start
        step_num = 1

    search_step, search_range, search_lower_bound = step_num, end - start, start

    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1.0, -1.0, -1.0)
    m_t = 0.0
    
    for i in range(search_step):
        threshold += search_range / float(search_step)
        # target, latency = calc_seq(score, label, threshold)
        target = calc_seq(score, label, threshold, K=K, calc_latency=False)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)

    return {
        "f1": m[0],
        "precision": m[1],
        "recall": m[2],
        "TP": m[3],
        "TN": m[4],
        "FP": m[5],
        "FN": m[6],
        "ROC_AUC" : m[7],
        "AUPRC" : m[8],
        "threshold": m_t
    }, m[9]

def valid_search(valid_score, score, label, start, end=None, interval=0.1, display_freq=1, K=0, verbose=True) -> object:
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """

    search_interval, search_range, search_lower_bound = interval, end - start, start

    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)

    threshold = search_lower_bound
    
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_range // search_interval):
        threshold = np.percentile(valid_score, 100-(i+1)*search_interval)
        target = calc_seq(score, label, threshold, K=K, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    return m, m_t



def epsilon_eval(train_scores, test_scores, test_labels, reg_level=1):
    # Global-level의 epsilon 구하기
    best_epsilon = find_epsilon(train_scores, reg_level)
    # pred, p_latency = adjust_predicts(test_scores, test_labels, best_epsilon, calc_latency=False)
    pred, p_latency = adjust_predicts2(test_scores, test_labels, best_epsilon, calc_latency=False, window=200)
    if test_labels is not None:
        p_t = calc_point2point(pred, test_labels)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "TP": p_t[3],
            "TN": p_t[4],
            "FP": p_t[5],
            "FN": p_t[6],
            "threshold": best_epsilon,
            # "latency": p_latency,
            "reg_level": reg_level,
        }
    else:
        return {"threshold": best_epsilon, "reg_level": reg_level}


def find_epsilon(errors, reg_level=1):
    """
    Threshold method proposed by Hundman et. al. (https://arxiv.org/abs/1802.04431)
    Code from TelemAnom (https://github.com/khundman/telemanom)
    """
    e_s = errors # train_pred_df[f"A_Score_{i}"].values (N - window,1)
    best_epsilon = None
    max_score = -10000000
    mean_e_s = np.mean(e_s) # Anomaly score 평균
    sd_e_s = np.std(e_s) # Anomaly score 표준편차

    # Z-score 기반 탐색 (z값을 2.5부터 12까지 0.5 간격으로 변경하며 여러 epsilon 후보를 생성)
    for z in np.arange(2.5, 12, 0.5): 
        epsilon = mean_e_s + sd_e_s * z # epsilon = 평균 + z × 표준편차
        pruned_e_s = e_s[e_s < epsilon] # epsilon보다 작은 점수들 (정상)

        # np.argwhere : 조건이 True인 인덱스를 반환
        i_anom = np.argwhere(e_s >= epsilon).reshape(-1,) # 이상치 후보 인덱스: epsilon 이상인 값들 (I,) (True인 idex 수)
        buffer = np.arange(1, 50) # 이상치로 판단된 index의 앞뒤로 포함시킬 거리 ( 1D numpy 배열 [1, 2, 3, ..., 49] )
       
        # i_anom : 이상치 본인 + 주변 49포인트 앞뒤 모두 포함된 index 집합
        i_anom = np.sort(
            np.concatenate(
                (
                    i_anom,
                    np.array([i + buffer for i in i_anom]).flatten(),
                    np.array([i - buffer for i in i_anom]).flatten(),
                )
            )
        )
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)] # index가 음수이거나 범위를 벗어나면 제거
        i_anom = np.sort(np.unique(i_anom)) # 중복된 인덱스 제거

        if len(i_anom) > 0:
            '''
            epsilon 이상인 값들을 이상치로 간주했을 때, 평균/표준편차가 얼마나 줄어드는지를 계산하고,
            이를 바탕으로 가장 의미 있는 이상치 분리 기준(best_epsilon)을 찾는 것.
            '''
            # 연속된 이상치 인덱스를 묶어 그룹화
            # EX] i_anom = [100, 101, 102, 200, 201] → groups = [[100,101,102], [200,201]]
            groups = [list(group) for group in mit.consecutive_groups(i_anom)]
            # E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

            # 통계량 변화율 계산 (평균/표준편차가 많이 줄었을수록 이상치를 잘 걸러냈다는 의미)
            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s # pruned_e_s : 정상이라고 간주된 데이터 (e_s < epsilon)
            sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s

            # 이상치가 너무 많을 경우 penalty 부여
            '''
            이상치로 분류된 값이 너무 많으면 오히려 잘못된 threshold일 가능성이 높기 때문에 점수에 패널티를 주는 것
            0: 규제 없음
            1: 이상치 수만큼 나눔 → 일반적인 사용
            2: 이상치 수 제곱으로 나눔 → 강하게 규제
            '''
            if reg_level == 0:
                denom = 1
            elif reg_level == 1:
                denom = len(i_anom)
            elif reg_level == 2:
                denom = len(i_anom) ** 2

            score = (mean_perc_decrease + sd_perc_decrease) / denom

            # 현재 score가 가장 높은 점수보다 크고, 이상치 수가 전체의 50%보다 적으면 best_epsilon으로 현재 epsilon 저장
            if score >= max_score and len(i_anom) < (len(e_s) * 0.5):
                max_score = score
                best_epsilon = epsilon
    
    # 모든 조건을 만족하지 못하면 fallback으로 max(e_s) 반환 (가장 큰 점수만 이상치로 간주하게 됨)
    if best_epsilon is None:
        best_epsilon = np.max(e_s)
    return best_epsilon


def pot_eval(init_score, score, label, q=1e-3, level=0.99, dynamic=False):
    """
    Run POT method on given score.
    :param init_score (np.ndarray): The data to get init threshold.
                    For `OmniAnomaly`, it should be the anomaly score of train set.
    :param: score (np.ndarray): The data to run POT method.
                    For `OmniAnomaly`, it should be the anomaly score of test set.
    :param label (np.ndarray): boolean list of true anomalies in score
    :param q (float): Detection level (risk)
    :param level (float): Probability associated with the initial threshold t
    :return dict: pot result dict
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """

    print(f"Running POT with q={q}, level={level}..")
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)
    s.initialize(level=level, min_extrema=False)  # Calibration step
    ret = s.run(dynamic=dynamic, with_alarm=False)

    print(len(ret["alarms"]))
    print(len(ret["thresholds"]))

    pot_th = np.mean(ret["thresholds"])
    # pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=False)
    pred, p_latency = adjust_predicts2(score, label, pot_th, calc_latency=False, window=200)
    if label is not None:
        p_t = calc_point2point(pred, label)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "TP": p_t[3],
            "TN": p_t[4],
            "FP": p_t[5],
            "FN": p_t[6],
            "threshold": pot_th,
            # "latency": p_latency,
        }
    else:
        return {
            "threshold": pot_th,
        }