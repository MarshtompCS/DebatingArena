import glob
import json
import os

DATA_PATH = "../data"


def load_topicchat_eval():
    score_file = os.path.join(DATA_PATH, "topic_chat/topical_chat.json")
    # with open(score_file, 'r', encoding='utf-8') as sf:
    #     topicchat_eval = [json.loads(line) for line in sf.readlines()]
    # return topicchat_eval
    return json.load(open(score_file, "r", encoding="utf-8"))


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
        src = open(src_path, 'r', encoding='utf-8').read().split()
        item["story"] = src

    return cnndailymail_eval


if __name__ == '__main__':
    # load_first_item()
    pass