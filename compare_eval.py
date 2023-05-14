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


def cnn_dailymail_compare_consistent(exp_name, resume=False, cur_exp_time=0):
    assert "cnn_dailymail" in exp_name
    assert "compare" in exp_name
    save_dir = f"./debate_results/{exp_name}_{cur_exp_time}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    cnn_dailymail_eval = load_cnndailymail_eval()
    cnn_dailymail_prompt = json.load(open("prompts/compare_cnn_consistent.json", "r", encoding="utf-8"))
    tqdm_bar = tqdm(cnn_dailymail_eval)
    for idx, item in enumerate(tqdm_bar):
        cur_save_path = os.path.join(save_dir, f"{item['idx']}.json")
        if resume and os.path.exists(cur_save_path) and not is_error_debate(load_json(cur_save_path)):
            continue
        global_prompt = cnn_dailymail_prompt["global_prompt"]
        global_prompt = global_prompt.format(
            cnn_dailymail_article=item["story"],
            cnn_dailymail_summary=item["decoded"],
        )
        cur_debate_arena = run_debate(item, cnn_dailymail_prompt, global_prompt,
                                      tqdm_bar=tqdm_bar, davinci_moderator=False)
        score_sentence = cur_debate_arena.environment.message_pool.last_message.content
        # save debate history and source info
        cur_history = cur_debate_arena.save_history(cur_save_path, return_dict=True)
        cur_history.append(item)
        json.dump(cur_history, open(cur_save_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)


def run_debate(item, system_prompts, global_prompt, tqdm_bar=None, davinci_moderator=True, max_debate_turns=6,
               max_tokens=512):
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

    player1 = Player(name="John", backend=OpenAIChat(**openai_kwargs),
                     role_desc=affirmative_player, global_prompt=global_prompt)
    player2 = Player(name="Eddy", backend=OpenAIChat(**openai_kwargs),
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
