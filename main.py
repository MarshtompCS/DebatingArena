import logging

import chatarena
from chatarena.agent import Moderator, Player, DebateModerator
from chatarena.backends import OpenAIChat
from chatarena.environments.conversation import Conversation, ModeratedConversation, ModeratedDebate
from chatarena.arena import Arena
import openai
from data_process import load_cnndailymail_eval, load_topicchat_eval
import json
from scipy import stats
from tqdm import tqdm
from collections import Counter
import re


def calculate_correlation_scores(data1, data2):
    spearmanr_res = stats.spearmanr(data1, data2)
    spearmanr_score, spearmanr_pvalue = spearmanr_res.statistic, spearmanr_res.pvalue
    kendalltau_res = stats.kendalltau(data1, data2)
    kendalltau_score, kendalltau_pvalue = kendalltau_res.statistic, kendalltau_res.pvalue
    return {"spearmanr_score": spearmanr_score, "spearmanr_pvalue": spearmanr_pvalue,
            "kendalltau_score": kendalltau_score, "kendalltau_pvalue": kendalltau_pvalue}


def get_score(sentence: str, min_valid=1, max_valid=3):  # find integer score
    score = None
    score_info = "error"
    if sentence.isdigit():
        score = int(sentence)
        if not (min_valid <= int(score) <= max_valid):
            score = None
            score_info = "certain"
    else:
        score_pattern = re.compile(r'\d+\.?\d*')
        candidates = score_pattern.findall(sentence)
        for candidate in reversed(candidates):
            if min_valid <= int(candidate) <= max_valid:
                score = int(candidate)
                score_info = "select last"
                break
    return score, score_info


def main():
    topicchat_eval = load_topicchat_eval()
    tqdm_bar = tqdm(topicchat_eval)
    predict_cnt = Counter()
    target_cnt = Counter()
    predict_scores, target_scores = [], []
    round_hits = 0
    for idx, item in enumerate(tqdm_bar):
        cur_debate_arena = run_debate(item)
        score_sentence = cur_debate_arena.environment.message_pool.last_message.content
        # predict score
        try:
            cur_predict_score, cur_score_info = get_score(score_sentence)
        except Exception as e:
            logging.error(f"idx={item['idx']}, score error: {e}")
            cur_predict_score = None
            cur_score_info = "error"
        predict_cnt.update([cur_predict_score])
        predict_scores.append(cur_predict_score)
        # human score
        cur_target_score = item["scores"]["engagingness"]
        target_scores.append(cur_target_score)
        target_cnt.update([round(cur_target_score)])
        if round(cur_target_score) == cur_predict_score:
            round_hits += 1
        bar_desc = f"round(target):{target_cnt.get(1, 0)}/{target_cnt.get(2, 0)}/{target_cnt.get(3, 0)} " \
                   f"predict:{predict_cnt.get(1, 0)}/{predict_cnt.get(2, 0)}/{predict_cnt.get(3, 0)} " \
                   f"errors:{predict_cnt.get(None, 0)} total/rhits:{idx + 1}/{round_hits}"
        tqdm_bar.set_description(bar_desc)

        # save debate history and source info
        cur_save_path = f"./debate_results/base_version/{item['idx']}.json"
        cur_history = cur_debate_arena.save_history(cur_save_path, return_dict=True)
        cur_history.append({
            "topic_chat_history": item["source"],
            "topic_chat_fact": item["context"],
            "topic_chat_response": item["system_output"],
            "human_score": cur_target_score,
            "predict_score": cur_predict_score,
            "predict_score_info": cur_score_info
        })
        json.dump(cur_history, open(cur_save_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

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
    json.dump(dump_info, open("./debate_results/base_version_info.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=4)


def run_debate(item, max_debate_turns=11, max_tokens=512):
    # 16 turns = 8 rounds, 9 turns = 4 rounds + 1, 12 + 1
    topic_chat_prompt = json.load(open("./topic_chat_prompt.json", "r", encoding="utf-8"))
    global_prompt = topic_chat_prompt["global_prompt"]
    global_prompt = global_prompt.format(
        topic_chat_history=item["source"],
        topic_chat_fact=item["context"],
        topic_chat_response=item["system_output"]
    )
    affirmative_player = topic_chat_prompt["affirmative_player"]
    negative_player = topic_chat_prompt["negative_player"]
    moderator_player = topic_chat_prompt["moderator_player"]
    terminate_prompt = topic_chat_prompt["terminate_prompt"]
    summarize_prompt = topic_chat_prompt["summarize_prompt"]
    engagingness_evaluation = topic_chat_prompt["engagingness_evaluation"]

    openai_kwargs = {"use_azure": True, "model": "gpt-35-turbo",
                     "max_tokens": max_tokens, "temperature": 0.0}

    player1 = Player(name="Affirmative", backend=OpenAIChat(**openai_kwargs),
                     role_desc=affirmative_player, global_prompt=global_prompt)
    player2 = Player(name="Negative", backend=OpenAIChat(**openai_kwargs),
                     role_desc=negative_player, global_prompt=global_prompt)
    moderator = DebateModerator(
        backend=OpenAIChat(**openai_kwargs), role_desc=moderator_player,
        terminal_condition=terminate_prompt, global_prompt=global_prompt,
        engagingness_evaluation=engagingness_evaluation, summarize_prompt=summarize_prompt
    )
    moderator_env = ModeratedDebate(
        player_names=["Affirmative", "Negative"], moderator=moderator,
        parallel=False, moderator_visibility=[],
        moderator_period="round", max_debate_turns=max_debate_turns
    )
    debate_arena = Arena(
        players=[player1, player2],
        environment=moderator_env,
        global_prompt=global_prompt
    )

    for step in range(16):
        # print(f"step-{step} start.")
        timestep = debate_arena.step()
        # print(f"step-{step} end.")
        if timestep.terminal:
            # print(f"Debate has terminated")
            break
    # print(f"target score is {item['scores']['engagingness']}")
    # print(moderator_env.message_pool.last_message.content)
    # debate_arena.save_history("./tmp.json")
    return debate_arena


if __name__ == '__main__':
    main()
"""
Global
这是一个评估对话系统的回复质量的辩论场，参与辩论的双方要根据“对话历史”，分别从正面和反面两个角度理性地评价给定“回复”。
回复的质量由一致性、流畅性、吸引性三个方面决定。
一致性表示：
流畅性表示：
吸引性表示：

Positive
你是正方，因此你要尽量找出这个“回复”的优点，并阐述理由。
你首先进行发言，随后你需要有理有据地反驳反方发言的不合理之处。

Negative
你是反方，因此你要尽量找出这个“回复”的缺点，并阐述理由。
正方先进行发言，你需要有理有据地反驳正方发言的不合理之处。

Moderator
你是这场辩论的裁判员。你需要公正地总结正方双反的发言并指出发言的合理和不合理之处。最后，你需要对“回复”的优劣进行打分，分数的取值范围是0-5。
辩论的总结：[hold]
回复的分数：
"""
