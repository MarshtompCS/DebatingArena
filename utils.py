from scipy import stats
import re
import logging
import json
import os


def is_valid_score(x):
    if type(x) not in [int, float]:
        return False
    return True


def calculate_correlation_scores(data1, data2):
    assert len(data1) == len(data2)
    # filter invalid data
    data = [(i, j) for i, j in zip(data1, data2) if is_valid_score(i) and is_valid_score(j)]
    if len(data) != len(data1):
        logging.warning(f"filtered invalid data points: {len(data1)} --> {len(data)}")
        data1 = [item[0] for item in data]
        data2 = [item[1] for item in data]
    spearmanr_res = stats.spearmanr(data1, data2)
    spearmanr_score, spearmanr_pvalue = spearmanr_res.statistic, spearmanr_res.pvalue
    kendalltau_res = stats.kendalltau(data1, data2)
    kendalltau_score, kendalltau_pvalue = kendalltau_res.statistic, kendalltau_res.pvalue
    pearsonr_res = stats.pearsonr(data1, data2)
    pearsonr_score, pearsonr_pvalue = pearsonr_res.statistic, pearsonr_res.pvalue
    return {"spearmanr_score": spearmanr_score, "spearmanr_pvalue": spearmanr_pvalue,
            "kendalltau_score": kendalltau_score, "kendalltau_pvalue": kendalltau_pvalue,
            "pearsonr_score": pearsonr_score, "pearsonr_pvalue": pearsonr_pvalue}


def get_score(sentence: str, min_valid=1, max_valid=3):  # find integer score
    score = None
    score_info = "error"
    if sentence.isnumeric():
        if min_valid <= int(sentence) <= max_valid:
            score = int(sentence)
            score_info = "certain"
    else:
        score_pattern = re.compile(r'\d+\.?\d*')
        candidates = score_pattern.findall(sentence)
        for candidate in candidates:
            if min_valid <= int(candidate) <= max_valid:
                score = int(candidate)
                score_info = "select first"
                break
    return score, score_info


def load_json(path):
    return json.load(open(path, "r", encoding="utf-8"))


def check_round_hits_num():
    # path = "./baseline_outputs/directly_score-text-davinci-003.json"  # 211
    path = "./baseline_outputs/with_cot_score-text-davinci-003.json"  # 194

    data = load_json(path)
    hitn = 0
    for i, j in zip(data["predict_scores"], data["target_scores"]):
        if i == round(j):
            hitn += 1
    print(hitn)


def system_rank():
    # data_dir = "./debate_results/davinci_moderator"
    # for i in range(360):
    #     cur_dir = os.path.join(data_dir, f"{i}.json")
    #     cur_data = load_json(cur_dir)

    data = load_json("./debate_results/davinci_moderator_info.json")

    scores_pred = [[] for _ in range(6)]
    scores_targ = [[] for _ in range(6)]

    for idx, (i, j) in enumerate(zip(data["predict_scores"], data["target_scores"])):
        scores_pred[idx % 6].append(i)
        scores_targ[idx % 6].append(j)

    scores_pred = [sum(i) / len(i) for i in scores_pred]
    scores_targ = [sum(i) / len(i) for i in scores_targ]

    print(scores_pred)
    print(scores_targ)


if __name__ == '__main__':
    # check_round_hits_num()
    system_rank()
