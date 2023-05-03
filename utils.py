from scipy import stats
import re
import logging
import json


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


if __name__ == '__main__':
    check_round_hits_num()
