import os
import json
import openai
import logging
from tqdm import tqdm
from tenacity import stop_after_attempt, retry, wait_random_exponential
from utils import get_score, calculate_correlation_scores
from collections import Counter

from data_process import load_topicchat_eval

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
STOP = ("<EOS>", "[EOS]", "(EOS)", "<|im_end|>")


@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
def get_response(model_name, messages, use_azure=True, temperature=0.0, max_tokens=256):
    if use_azure:
        completion = openai.ChatCompletion.create(
            engine=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=STOP
        )
    else:
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=STOP
        )
    response = completion.choices[0]['message']['content']
    response = response.strip()
    return response


@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
def get_completion(model_name, prompt, use_azure=True, temperature=0.0, max_tokens=256):
    if use_azure:
        completion = openai.Completion.create(
            engine=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=STOP
        )
    else:
        completion = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=STOP
        )
    response = completion["choices"][0]["text"]
    response = response.strip()
    return response


def set_env():
    config = json.load(open("./env_config.json", 'r', encoding='utf-8'))

    if config["use_openai"]:
        if config["use_azure"]:
            azure_openai_config = config["azure_openai"]
            openai.api_type = azure_openai_config["api_type"]
            openai.api_base = azure_openai_config["api_base"]
            openai.api_version = azure_openai_config["ChatCompletion_api_version"]
            openai.api_key = azure_openai_config["api_key"]
        else:
            openai_config = config["openai"]
            openai.api_key = openai_config["api_key"]


def g_eval(with_cot=False, with_weighted=False):
    topicchat_eval = load_topicchat_eval()
    tqdm_bar = tqdm(topicchat_eval)

    if with_cot:
        origin_prompt = json.load(open("./baseline_prompts.json", "r", encoding="utf-8"))["with_cot_score"]
    else:
        origin_prompt = json.load(open("./baseline_prompts.json", "r", encoding="utf-8"))["directly_score"]

    target_scores, predict_scores = [], []
    completions = []
    predict_cnt = Counter()
    target_cnt = Counter()
    round_hits = 0
    for idx, item in enumerate(tqdm_bar):
        cur_prompt = origin_prompt.format(
            topic_chat_history=item["source"],
            topic_chat_fact=item["context"],
            topic_chat_response=item["system_output"]
        )
        cur_completion = get_completion(model_name="text-davinci-003", prompt=cur_prompt)

        try:
            cur_predict_score, cur_score_info = get_score(cur_completion)
        except Exception as e:
            logging.error(f"idx={item['idx']}, score error: {e}")
            cur_predict_score = None
            cur_score_info = "error"
        cur_target_score = item["scores"]["engagingness"]
        completions.append(cur_completion)
        target_scores.append(cur_target_score)
        predict_scores.append(cur_predict_score)
        predict_cnt.update([cur_predict_score])
        target_cnt.update([round(cur_target_score)])
        if round(cur_target_score) == cur_predict_score:
            round_hits += 1
        bar_desc = f"round(target):{target_cnt.get(1, 0)}/{target_cnt.get(2, 0)}/{target_cnt.get(3, 0)} " \
                   f"predict:{predict_cnt.get(1, 0)}/{predict_cnt.get(2, 0)}/{predict_cnt.get(3, 0)} " \
                   f"errors:{predict_cnt.get(None, 0)} total/rhits:{idx + 1}/{round_hits}"
        tqdm_bar.set_description(bar_desc)

    correlation_results = calculate_correlation_scores(predict_scores, target_scores)
    dump_info = {
        "predict_scores": predict_scores,
        "target_scores": target_scores,
        "predict_cnt": {
            1: predict_cnt.get(1, 0),
            2: predict_cnt.get(2, 0),
            3: predict_cnt.get(3, 0),
            "error": predict_cnt.get(None, 0)
        },
        "target_round_cnt": {
            1: target_cnt.get(1, 0),
            2: target_cnt.get(2, 0),
            3: target_cnt.get(3, 0),
        }
    }
    dump_info.update(correlation_results)
    if with_cot:
        dump_path = "./baseline_outputs/with_cot_score-text-davinci-003.json"
    else:
        dump_path = "./baseline_outputs/directly_score-text-davinci-003.json"

    json.dump(dump_info, open(dump_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    set_env()

    # g_eval()
    g_eval(with_cot=True)
