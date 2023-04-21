import chatarena
from chatarena.agent import Player
from chatarena.backends import OpenAIChat
from chatarena.environments.conversation import Conversation
from chatarena.arena import Arena
import openai

arena = Arena.from_config("debate_examples/azure_test.json")
for _ in range(2):
    timestep = arena.step()
    msg = timestep.observation[-1]
    print("-" * 100)
    print(f"{msg.agent_name}: {msg.content}")
    print("-" * 100)

arena.save_history("debate_examples/output2.json")
