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


def is_error_debate(debate_history):
    for msg in debate_history[:-1]:
        if msg["content"].startswith("<<<<<<END_OF_CONVERSATION>>>>>>"):
            return True
    return False


def cnn_dailymail_check():
    dir_path = "./debate_results/cnn_dailymail_base_version"
    err_indexes = []
    for idx in range(92):
        cur_path = os.path.join(dir_path, f"{idx}.json")
        cur_data = load_json(cur_path)
        if is_error_debate(cur_data):
            err_indexes.append(idx)
            break
    print(err_indexes)


def get_cnn_dailymail_human_scores(item):
    expert_annotations = item["expert_annotations"]
    turker_annotations = item["turker_annotations"]
    form_keys = ["Relevance", "Consistency", "Fluency", "Coherence"]

    expert_eval = {k: sum([expert[k.lower()] for expert in expert_annotations]) /
                      len(expert_annotations) for k in form_keys}

    turker_eval = {k: sum([turker[k.lower()] for turker in turker_annotations]) /
                      len(turker_annotations) for k in form_keys}

    all_human_eval = {k: sum([human[k.lower()] for human in expert_annotations + turker_annotations]) /
                         len(expert_annotations + turker_annotations) for k in form_keys}

    return {"expert_eval": expert_eval,
            "turker_eval": turker_eval,
            "all_human_eval": all_human_eval}


def parse_cnn_dailymail_scores(debates_dir, debates_num=1700):
    all_pred = []
    all_human = []
    form_keys = ["Relevance", "Consistency", "Fluency", "Coherence"]
    for idx in range(debates_num):
        cur_path = os.path.join(debates_dir, f"{idx}.json")
        cur_debate = load_json(cur_path)

        cur_human = get_cnn_dailymail_human_scores(cur_debate[-1])
        all_human.append(cur_human)

        if is_error_debate(cur_debate):
            all_pred.append(None)
            continue

        cur_eval_form = cur_debate[-2]
        cur_lines = cur_eval_form.contend.split("\n")
        cur_pred = {k: None for k in form_keys}

        try:
            cur_pred["Relevance"] = int(cur_lines[0])
        except ValueError as err:
            logging.error(f"error: {err}\nparse {cur_lines[0]}")

        for line in cur_lines[1:4]:
            try:
                contain_keys = [k for k in form_keys if k in line]
                assert len(contain_keys) == 1
                assert cur_pred[contain_keys[0]] is None
                score_pattern = re.compile(r'\d+\.?\d*')
                candidates = score_pattern.findall(line)
                assert len(candidates) == 1
            except AssertionError as err:
                logging.error(f"error: {err}\nparse line: {line}")
                break
            cur_pred[contain_keys[0]] = int(candidates[0])

        all_pred.append(cur_pred)

    return all_pred, all_human


def filtered_two_list(data1, data2):
    all_data = [(i, j) for i, j in zip(data1, data2)
                if i is not None and j is not None]
    data1 = [i[0] for i in all_data]
    data2 = [i[1] for i in all_data]
    return data1, data2


def cnn_dailymail_correlation(all_human, all_pred):
    all_human, all_pred = filtered_two_list(all_human, all_pred)

    form_keys = ["Relevance", "Consistency", "Fluency", "Coherence"]
    for k in form_keys:
        print(f"{k} correlation")

        print(f"{k} expert correlation")
        expert_k = [item["expert"][k] for item in all_human]
        pred_k = [item[k] for item in all_pred]
        expert_k, pred_k = filtered_two_list(expert_k, pred_k)
        expert_res = calculate_correlation_scores(expert_k, pred_k)
        print(expert_res)

        print(f"{k} turker correlation")
        turker_k = [item["turker"][k] for item in all_human]
        pred_k = [item[k] for item in all_pred]
        turker_k, pred_k = filtered_two_list(turker_k, pred_k)
        turker_res = calculate_correlation_scores(turker_k, pred_k)
        print(turker_res)

        print(f"{k} all_human correlation")
        human_k = [item["all_human"][k] for item in all_human]
        pred_k = [item[k] for item in all_pred]
        human_k, pred_k = filtered_two_list(human_k, pred_k)
        human_res = calculate_correlation_scores(human_k, pred_k)
        print(human_res)


def multi_debate_cnn_dailymail():
    dirs = []
    all_debate_pred = []
    all_human = []
    for cur_dir in dirs:
        cur_debate_pred, all_human = parse_cnn_dailymail_scores(cur_dir, debates_num=1700)
        print(f"\n{cur_dir} correlation")
        cnn_dailymail_correlation(all_human, cur_debate_pred)
        all_debate_pred.append(cur_debate_pred)

    ensemble_pred = []
    form_keys = ["Relevance", "Consistency", "Fluency", "Coherence"]
    for idx in range(1700):
        cur_preds = [debates[idx] for debates in all_debate_pred]
        if all(x is None for x in cur_preds):
            ensemble_pred.append(None)
            continue

        cur_ensemble_pred = {k: None for k in form_keys}
        for k in form_keys:
            cur_pred_k = [cur_pred[k] for cur_pred in cur_preds if cur_pred is not None]
            if len(cur_pred_k) > 0:
                cur_pred_k = sum(cur_pred_k) / len(cur_pred_k)
            else:
                cur_pred_k = None
            cur_ensemble_pred[k] = cur_pred_k

        ensemble_pred.append(cur_ensemble_pred)

    print("\nensemble correlation")
    cnn_dailymail_correlation(all_human, ensemble_pred)


if __name__ == '__main__':
    # check_round_hits_num()
    # system_rank()
    cnn_dailymail_check()
