import chatarena
from chatarena.agent import Moderator, Player, DebateModerator
from chatarena.backends import OpenAIChat
from chatarena.environments.conversation import Conversation, ModeratedConversation, ModeratedDebate
from chatarena.arena import Arena
import openai
from data_process import load_cnndailymail_eval, load_topicchat_eval
import json


# TODO
# add instructions: do not repeat, be concise, ...

def main():
    item = load_topicchat_eval()[0]
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
    player1 = Player(name="affirmative", backend=OpenAIChat(use_azure=True, model="gpt-35-turbo"),
                     role_desc=affirmative_player, global_prompt=global_prompt)
    player2 = Player(name="negative", backend=OpenAIChat(use_azure=True, model="gpt-35-turbo"),
                     role_desc=negative_player, global_prompt=global_prompt)
    moderator = DebateModerator(backend=OpenAIChat(use_azure=True, model="gpt-35-turbo"), role_desc=moderator_player,
                                terminal_condition=terminate_prompt, global_prompt=global_prompt,
                                engagingness_evaluation=engagingness_evaluation, summarize_prompt=summarize_prompt)
    moderator_env = ModeratedDebate(
        player_names=["affirmative", "negative"],
        moderator=moderator, parallel=False,
        moderator_visibility=[], moderator_period="round", max_debate_turns=6
    )

    debate_arena = Arena(players=[player1, player2],
                         environment=moderator_env,
                         global_prompt=global_prompt)

    for step in range(16):
        print(f"step-{step} start.")
        timestep = debate_arena.step()
        print(f"step-{step} end.")
        if timestep.terminal:
            break

    debate_arena.save_history("./tmp.json")


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
