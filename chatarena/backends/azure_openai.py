from typing import List
import os
import re
import logging
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import IntelligenceBackend
from ..message import Message

try:
    import openai
except ImportError:
    is_openai_available = False
    logging.warning("openai package is not installed")
else:
    openai.api_key = os.environ.get("OPENAI_API_KEY", openai.api_key)
    if openai.api_key is None:
        logging.warning("OpenAI API key is not set. Please set the environment variable OPENAI_API_KEY")
        is_openai_available = False
    else:
        is_openai_available = True

# Default config follows the OpenAI playground
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "gpt-35-turbo"  # it is equal to "gpt-3.5-turbo" used in OpenAI API

DEFAULT_API = "ChatCompletion"

STOP = ("<EOS>", "[EOS]", "(EOS)", "<|im_end|>")  # End of sentence token
# STOP = ("<|im_end|>",) # default eos token introduced in azure-openai document

BOS = "<|im_start|>"
EOS = "<|im_end|>"


class AzureOpenAIChat(IntelligenceBackend):
    """
    Interface to the ChatGPT style model with system, user, assistant roles separation
    """
    stateful = False
    type_name = "azure-openai-chat"

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS,
                 model: str = DEFAULT_MODEL, api_name: str = DEFAULT_API, **kwargs):
        assert is_openai_available, "openai package is not installed or the API key is not set"
        super().__init__(temperature=temperature, max_tokens=max_tokens, model=model, **kwargs)

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model

        assert api_name in ["ChatCompletion", "Completion"]
        self.api_name = api_name

        openai.api_version = "2022-12-01" if api_name == "Completion" else "2023-03-15-preview"


    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response_completion(self, prompt):
        completion = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=STOP
        )
        response = completion["choices"][0]["text"]
        response = response.strip()
        return response

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response_chat_completion(self, messages):
        completion = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=STOP
        )
        response = completion['choices'][0]['message']['content']
        response = response.strip()
        return response

    def _create_input_azure_chat_completion(self, agent_name: str, role_desc: str, history_messages: List[Message],
                                           global_prompt: str = None, request_msg: Message = None) -> list:
        conversations = []
        for i, message in enumerate(history_messages):
            if message.agent_name == agent_name:
                conversations.append({"role": "assistant", "content": message.content})
            else:
                # Since there are more than one player, we need to distinguish between the players
                conversations.append({"role": "user", "content": f"[{message.agent_name}]: {message.content}"})

        if global_prompt:  # Prepend the global prompt if it exists
            system_prompt_str = f"{global_prompt.strip()}\n{role_desc}"
        else:
            system_prompt_str = role_desc
        system_prompt = {"role": "system", "content": system_prompt_str}

        if request_msg:
            request_prompt = [{"role": "user", "content": request_msg.content}]
        else:
            request_prompt = []

        return [system_prompt] + conversations + request_prompt

    def _create_input_azure_completion(self, agent_name: str, role_desc: str, history_messages: List[Message],
                                      global_prompt: str = None, request_msg: Message = None) -> str:
        if global_prompt:  # Prepend the global prompt if it exists
            system_prompt_str = f"{global_prompt.strip()}\n{role_desc}"
        else:
            system_prompt_str = role_desc

        str_prompt = f"{BOS}system\n{system_prompt_str}\n{EOS}"

        for i, message in enumerate(history_messages):
            if message.agent_name == agent_name:
                msg_prompt = f"{BOS}assistant\n{message.content}\n{EOS}"
            else:
                msg_prompt = f"{BOS}user\n[{message.agent_name}]: {message.content}\n{EOS}"
            str_prompt += msg_prompt

        if request_msg:
            str_prompt += f"{BOS}user\n[{request_msg.agent_name}]: {request_msg.content}\n{EOS}"

        str_prompt += f"{BOS}assistant\n"

        return str_prompt

    def query(self, agent_name: str, role_desc: str, history_messages: List[Message], global_prompt: str = None,
              request_msg: Message = None, *args, **kwargs) -> str:
        """
        format the input and call the ChatGPT/GPT-4 API
        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            env_desc: the description of the environment
            history_messages: the history of the conversation, or the observation for the agent
            request_msg: the request for the chatGPT
        """

        if self.api_name == "ChatCompletion":
            messages = self._create_input_azure_chat_completion(
                agent_name, role_desc, history_messages, global_prompt, request_msg
            )
            response = self._get_response_chat_completion(messages, *args, **kwargs)
        else:
            logging.warning("You're using Completion api")
            prompt = self._create_input_azure_completion(
                agent_name, role_desc, history_messages, global_prompt, request_msg
            )
            response = self._get_response_completion(prompt, *args, **kwargs)

        # Remove the agent name if the response starts with it
        response = re.sub(rf"^\s*\[{agent_name}]:", "", response)
        response = response.strip()
        return response
