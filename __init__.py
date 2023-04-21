import os
import json
import openai


# import os
# os.environ["OPENAI_API_KEY"] = "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"


def azure_debug():
    openai.api_type = "azure"
    openai.api_base = "https://ai-lab-openai.openai.azure.com/"
    openai.api_version = "2022-12-01"
    openai.api_key = "45769099f37a4c02b7b3f79484acf347"

    global_prompt = "You are in a university classroom and it is Natural Language Processing module. You start by introducing themselves. Your answer will end with <EOS>."
    role_prompt = "You are Prof. Alpha, a knowledgeable professor in NLP. Your answer will concise and accurate. The answers should be less than 100 words."

    system_prompt = f"{global_prompt.strip()}\n{role_prompt}"
    messages = [{"role": "system", "content": system_prompt}]
    # messages += [{"role": "User", "content": "Hello, I am a NLP professor. Do you have NLP research background?"}]

    bos = "<|im_start|>"
    eos = "<|im_end|>"
    prompt = ""
    for msg in messages:
        prompt += f"{bos}{msg['role']}\n{msg['content']}\n{eos}"
    prompt += f"{bos}assistant\n"

    max_tokens = 256
    top_p = 0.9
    STOP = ("<EOS>", "[EOS]", "(EOS)")  # End of sentence token
    # ["<|im_end|>"]
    response = openai.Completion.create(
        engine="gpt-35-turbo",  # The deployment name you chose when you deployed the ChatGPT model
        # model="gpt-3.5-turbo",
        prompt=prompt,
        # messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=STOP
    )
    print(response)

    system_message = {"role": "system", "content": "You are a helpful assistant."}
    max_response_tokens = 250
    conversation = []
    conversation.append(system_message)
    conversation.append({"role": "user", "content": "hello"})
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",  # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
        messages=conversation,
        temperature=.7,
        max_tokens=max_response_tokens,
    )
    print(response)


azure_debug()
