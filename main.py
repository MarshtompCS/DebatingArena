import logging
import chatarena
from chatarena.agent import Moderator, Player, DebateModerator
from chatarena.backends import OpenAIChat
from chatarena.backends.openai import OpenAICompletion
from chatarena.environments.conversation import Conversation, ModeratedConversation, ModeratedDebate
from chatarena.arena import Arena
import openai
from data_process import load_cnndailymail_eval, load_topicchat_eval
import json
from tqdm import tqdm
from collections import Counter
from utils import calculate_correlation_scores, get_score, load_json, is_error_debate
import os


def topical_chat_engagingness(exp_name):
    if not os.path.exists(f"./debate_results/{exp_name}"):
        os.mkdir(f"./debate_results/{exp_name}")

    topicchat_eval = load_topicchat_eval()
    topic_chat_prompt = json.load(open("./topic_chat_prompt.json", "r", encoding="utf-8"))
    tqdm_bar = tqdm(topicchat_eval)
    predict_cnt = Counter()
    target_cnt = Counter()
    predict_scores, target_scores = [], []
    round_hits = 0
    for idx, item in enumerate(tqdm_bar):
        global_prompt = topic_chat_prompt["global_prompt"]
        global_prompt = global_prompt.format(
            topic_chat_history=item["source"],
            topic_chat_fact=item["context"],
            topic_chat_response=item["system_output"]
        )
        cur_debate_arena = run_debate(item, topic_chat_prompt, global_prompt)
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
        cur_save_path = f"./debate_results/{exp_name}/{item['idx']}.json"
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
    json.dump(dump_info, open(f"./debate_results/{exp_name}_info.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=4)


def cnn_dailymail_four_aspects(exp_name, resume=False, cur_exp_time=0):
    assert "cnn_dailymail" in exp_name
    save_dir = f"./debate_results/{exp_name}_{cur_exp_time}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    cnn_dailymail_eval = load_cnndailymail_eval()
    cnn_dailymail_prompt = json.load(open("./cnn_dailymail_prompt.json", "r", encoding="utf-8"))
    tqdm_bar = tqdm(cnn_dailymail_eval)
    # predict_cnt = Counter()
    # target_cnt = Counter()
    # predict_scores, target_scores = [], []
    # round_hits = 0
    for idx, item in enumerate(tqdm_bar):
        cur_save_path = os.path.join(save_dir, f"{item['idx']}.json")

        if resume and os.path.exists(cur_save_path) and not is_error_debate(load_json(cur_save_path)):
            continue

        global_prompt = cnn_dailymail_prompt["global_prompt"]
        global_prompt = global_prompt.format(
            cnn_dailymail_article=item["story"],
            cnn_dailymail_summary=item["decoded"],
        )
        cur_debate_arena = run_debate(item, cnn_dailymail_prompt, global_prompt, tqdm_bar=tqdm_bar)
        score_sentence = cur_debate_arena.environment.message_pool.last_message.content
        # predict score
        # try:
        #     cur_predict_score, cur_score_info = get_score(score_sentence)
        # except Exception as e:
        #     logging.error(f"idx={item['idx']}, score error: {e}")
        #     cur_predict_score = None
        #     cur_score_info = "error"
        # predict_cnt.update([cur_predict_score])
        # predict_scores.append(cur_predict_score)
        # human score
        # cur_target_score = item["scores"]["engagingness"]
        # target_scores.append(cur_target_score)
        # target_cnt.update([round(cur_target_score)])
        # if round(cur_target_score) == cur_predict_score:
        #     round_hits += 1
        # bar_desc = f"round(target):{target_cnt.get(1, 0)}/{target_cnt.get(2, 0)}/{target_cnt.get(3, 0)} " \
        #            f"predict:{predict_cnt.get(1, 0)}/{predict_cnt.get(2, 0)}/{predict_cnt.get(3, 0)} " \
        #            f"errors:{predict_cnt.get(None, 0)} total/rhits:{idx + 1}/{round_hits}"
        # tqdm_bar.set_description(bar_desc)

        # save debate history and source info
        cur_history = cur_debate_arena.save_history(cur_save_path, return_dict=True)
        cur_history.append(item)
        # ({
        # "article": item["source"],
        # "summary": item["context"],
        # "human_score": cur_target_score,
        # "predict_score": cur_predict_score,
        # "predict_score_info": cur_score_info
        # })
        json.dump(cur_history, open(cur_save_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

    # correlation_results = calculate_correlation_scores(predict_scores, target_scores)
    # dump_info = {
    #     "predict_scores": predict_scores,
    #     "target_scores": target_scores,
    #     "predict_cnt": {
    #         1: predict_cnt.get(1, 0),
    #         2: predict_cnt.get(2, 0),
    #         3: predict_cnt.get(3, 0),
    #         "error": predict_cnt.get(None, 0)
    #     },
    #     "target_round_cnt": {
    #         1: target_cnt.get(1, 0),
    #         2: target_cnt.get(2, 0),
    #         3: target_cnt.get(3, 0),
    #     }
    # }
    # dump_info.update(correlation_results)
    # json.dump(dump_info, open(f"./debate_results/{exp_name}_info.json", "w", encoding="utf-8"),
    #           ensure_ascii=False, indent=4)


def run_debate(item, system_prompts, global_prompt, tqdm_bar=None, davinci_moderator=True, max_debate_turns=6,
               max_tokens=512):
    # 16 turns = 8 rounds, 9 turns = 4 rounds + 1, 12 + 1
    # topic_chat_prompt = json.load(open("./topic_chat_prompt.json", "r", encoding="utf-8"))

    affirmative_player = system_prompts["affirmative_player"]
    negative_player = system_prompts["negative_player"]
    moderator_player = system_prompts["moderator_player"]
    terminate_prompt = system_prompts["terminate_prompt"]
    summarize_prompt = system_prompts["summarize_prompt"]
    evaluation_prompt = system_prompts["evaluation_prompt"]
    if "{" in evaluation_prompt:
        evaluation_prompt = evaluation_prompt.format(
            topic_chat_response=item["system_output"]
        )
    openai_kwargs = {"use_azure": True, "model": "gpt-35-turbo",
                     "max_tokens": max_tokens, "temperature": 0.5}

    player1 = Player(name="Affirmative", backend=OpenAIChat(**openai_kwargs),
                     role_desc=affirmative_player, global_prompt=global_prompt)
    player2 = Player(name="Negative", backend=OpenAIChat(**openai_kwargs),
                     role_desc=negative_player, global_prompt=global_prompt)

    if davinci_moderator:
        moderator_backend = OpenAICompletion(use_azure=True, model="text-davinci-003",
                                             max_tokens=max_tokens, temperature=0.0)
        completion_prefix = system_prompts["completion_prefix"]
    else:
        openai_kwargs["temperature"] = 0.0
        moderator_backend = OpenAIChat(**openai_kwargs)
        completion_prefix = None

    moderator = DebateModerator(backend=moderator_backend, role_desc=moderator_player,
                                terminal_condition=terminate_prompt, global_prompt=global_prompt,
                                evaluation_prompt=evaluation_prompt, summarize_prompt=summarize_prompt,
                                completion_prefix=completion_prefix)
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
        tqdm_bar.set_description(f"debate step: {step}")
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


def run_comparison_debate():
    # text summarization
    pass


if __name__ == '__main__':
    # main("davinci_moderator")
    for exp_time in range(5):
        cnn_dailymail_four_aspects("cnn_dailymail_base_version", resume=True, cur_exp_time=exp_time)
