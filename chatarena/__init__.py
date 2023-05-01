import os
import json
import openai
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


#
#
# def set_env():
#     config = json.load(open("./env_config.json", 'r', encoding='utf-8'))
#     logging.info(f"load config successfully")
#
#     if config["use_openai"]:
#         if config["use_azure"]:
#             azure_openai_config = config["azure_openai"]
#             openai.api_type = azure_openai_config["api_type"]
#             openai.api_base = azure_openai_config["api_base"]
#             # openai.api_version = azure_openai_config["api_version"]  # moved to azure.agent
#             os.environ["OPENAI_API_KEY"] = azure_openai_config["api_key"]
#             logging.info(f"azure-openai set")
#         else:
#
#             openai_config = config["openai"]
#             openai.api_key = openai_config["api_key"]
#             os.environ["OPENAI_API_KEY"] = openai_config["api_key"]
#             logging.info(f"openai key set")
#
#
# set_env()

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


set_env()
