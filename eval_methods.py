import numpy as np
import more_itertools as mit
from spot import SPOT, dSPOT


def adjust_predicts(score, label, threshold, pred=None, calc_latency=False):

    if label is None:
        predict = score > threshold
        return predict, None

    # adjust predict
    if pred is None:
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        predict = score > threshold
    else:
        predict = pred

    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    latency = 0
    for i in range(len(predict)):
        # actual[max(i,0) : i+1] 其实就是actual[i]，因为i始终是正整数
        if any(actual[max(i, 0) : i + 1]) and predict[i] and not anomaly_state: # 如果真实的是true且预测是true且异常状态是false，即第一次检测到故障
            # 说明在i之前，actual已经出现了true
            anomaly_state = True
            anomaly_count += 1
            # 遍历actual[i]到actual[0]，计算从故障发生到检测出故障所用的延迟latency
            # 并且将延迟期间的预测结果都改为true
            for j in range(i, 0, -1): #j不包括0
                # 若actual[j]是false
                if not actual[j]:
                    break
                # 若actual[j]是true
                else:
                    # 若predict[j]是false
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:# 如果真实是false
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        # 返回predict和平均每个故障从发生到检测出来的延迟时间步
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
            predict (np.ndarray): the predict label
            actual (np.ndarray): np.ndarray
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    # predict是true且actual是true的个数，被预测为故障的真故障个数
    actual = actual.reshape(-1)
    TP = np.sum(predict * actual)
    # predict是false且actual是false的个数，被预测为无故障的无故障个数
    TN = np.sum((1 - predict) * (1 - actual))
    # predict是true且actual是false，被预测为故障的无故障个数
    FP = np.sum(predict * (1 - actual))
    # predict是false且actual是true，被预测为无故障的有故障个数
    FN = np.sum((1 - predict) * actual) #经过adjust_predicts函数，FN一定为0
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    print("Accuracy=%f" % ((TP) / len(predict)))
    print("================================================")
    return f1, precision, recall, TP, TN, FP, FN


def pot_eval(init_score, score, label, q=1e-3, level=0.99, dynamic=False):
    print(f"Running POT with q={q}, level={level}..")
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)
    s.initialize(level=level, min_extrema=False)  # Calibration step
    ret = s.run(dynamic=dynamic, with_alarm=False)

    print(len(ret["alarms"]))
    print(len(ret["thresholds"]))

    pot_th = np.mean(ret["thresholds"])  # POT算出来的threshold
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
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
            "latency": p_latency,
        }
    else:
        return {
            "threshold": pot_th,
        }

def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
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
    m_l = 0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target, latency = adjust_predicts(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
            m_l = latency
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
        "threshold": m_t,
        "latency": m_l,
    }

def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def epsilon_eval(train_scores, test_scores, test_labels, reg_level=1):
    """
    train_scores:df["A_Global_Scores"]=(len(train),) 对每一行的score求相加求平均
    test_scores:df["A_Global_Scores"]=(len(test),)
    """

    best_epsilon = find_epsilon(train_scores, reg_level)  # 根据均值、方差等找到最优阈值
    # best_epsilon是所有features平均值的score的阈值。
    pred, p_latency = adjust_predicts(test_scores, test_labels, best_epsilon, calc_latency=True)  # 修正preds

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
            "latency": p_latency,
            "reg_level": reg_level,
        }
    else:
        return {"threshold": best_epsilon, "reg_level": reg_level}

def find_epsilon(errors, reg_level=1):
    """
    Threshold method proposed by Hundman et. al. (https://arxiv.org/abs/1802.04431)
    Code from TelemAnom (https://github.com/khundman/telemanom)
    """
    e_s = errors
    best_epsilon = None
    max_score = -10000000
    mean_e_s = np.mean(e_s)
    sd_e_s = np.std(e_s)  # 标准差

    for z in np.arange(2.5, 12, 0.5):
        epsilon = mean_e_s + sd_e_s * z  # 寻找使得score最大的epsilon
        pruned_e_s = e_s[e_s < epsilon]

        i_anom = np.argwhere(e_s >= epsilon).reshape(-1, )  # 变成1列。所有>epsilon的索引？
        buffer = np.arange(1, 50)
        i_anom = np.sort(
            np.concatenate(
                (
                    i_anom,
                    np.array([i + buffer for i in i_anom]).flatten(),
                    np.array([i - buffer for i in i_anom]).flatten(),
                )
            )
        )
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        if len(i_anom) > 0:
            groups = [list(group) for group in mit.consecutive_groups(i_anom)]  # 标识连续数组
            # E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]] 源代码这里有注释
            E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]
            E_esq_2 = len(E_seq) ** 2
            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s
            sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s
            if reg_level == 0:
                denom = 1
            elif reg_level == 1:
                denom = len(i_anom)
            elif reg_level == 2:
                denom = len(i_anom) ** 2

            # score = (mean_perc_decrease + sd_perc_decrease) / (denom)
            score = (mean_perc_decrease + sd_perc_decrease) / (denom + E_esq_2)

            if score >= max_score and len(i_anom) < (len(e_s) * 0.5):
                max_score = score
                best_epsilon = epsilon  # 寻找使得max_score最大的epsilon

    if best_epsilon is None:
        best_epsilon = np.max(e_s)
    return best_epsilon

