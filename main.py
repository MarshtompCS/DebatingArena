import chatarena
from chatarena.agent import Player
from chatarena.backends import OpenAIChat
from chatarena.environments.conversation import Conversation
from chatarena.arena import Arena
import openai


def main():
    global_prompt = "You are in a debating competition to judge whether a response is good for a question. \n" \
                    "Here is the question: \n{question}\n\n" \
                    "Here is the response: \n{response}\n\n" \
                    "There are one positive speaker, one negative speaker, and one moderator. " \
                    "Negative and positive speakers speak in turn. " \
                    "The moderator will decide when to stop the debating."

    player1 = Player(
        name="Bob",
        backend=OpenAIChat(),
        role_desc="You are positive, "
                  "so you should do your best to find the advantages of the response and "
                  "explain the reasons in this debate. You are the first to speak, "
                  "and then you should rebut the opinions from the con side.",
        global_prompt=global_prompt,
    )
    player2 = Player(
        name="Alice",
        backend=OpenAIChat(),
        role_desc="You are negative, "
                  "so you should do your best to find the disadvantages of the response and "
                  "explain the reasons in this debate. "
                  "You are the second to speak, and you should rebut the opinions from the pro side.",
        global_prompt=global_prompt,
    )
    player3 = Player(
        name="Moderator",
        backend=OpenAIChat(),
        role_desc="You are the moderator of the debate. "
                  "You should judge the rationality and correctness of the speaking from both side. "
                  "You should stop the debate when either side have no chance to rebut the other side. "
                  "Finally, you should summarize the debating, and score the response",
        global_prompt=global_prompt,
    )

    # Environment
    environment = Conversation()


if __name__ == '__main__':
    main()

# arena = Arena.from_config("debate_examples/azure_test.json")
# for _ in range(2):
#     timestep = arena.step()
#     msg = timestep.observation[-1]
#     print("-" * 100)
#     print(f"{msg.agent_name}: {msg.content}")
#     print("-" * 100)
#
# arena.save_history("debate_examples/output2.json")
