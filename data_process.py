import glob
import json
import os

DATA_PATH = "../data"


def load_topicchat_eval():
    score_file = os.path.join(DATA_PATH, "topic_chat/topical_chat.json")
    # with open(score_file, 'r', encoding='utf-8') as sf:
    #     topicchat_eval = [json.loads(line) for line in sf.readlines()]
    # return topicchat_eval
    data = json.load(open(score_file, "r", encoding="utf-8"))
    if "idx" not in data[0]:
        for idx, item in enumerate(data):
            item["idx"] = idx
        json.dump(data, open(score_file, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    return data


def load_cnndailymail_eval():
    score_file = os.path.join(DATA_PATH, "model_annotations.aligned.scored.jsonl")
    with open(score_file, "r", encoding="utf-8") as sf:
        cnndailymail_eval = [json.loads(line) for line in sf.readlines()]

    for item in cnndailymail_eval:
        dataset_name, dataset_split, cur_src_id = item["id"].split("-")
        assert dataset_split == "test"
        if dataset_name == "dm":
            dataset_name = "dailymail"
        src_path = os.path.join(DATA_PATH, f"{dataset_name}/stories/{cur_src_id}.story")
        src = open(src_path, 'r', encoding='utf-8').read().strip()
        item["story"] = src

    return cnndailymail_eval


if __name__ == '__main__':
    # load_first_item()
    load_cnndailymail_eval()
    pass
