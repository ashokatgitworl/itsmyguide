from paths import CHROMA_URLL,CHROMA_PORTT,ROOT_DIR
from utils import load_yaml_config
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR
from prompt_builder import build_prompt_from_config
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR
from utils import load_yaml_config
   


# print("ROOT_DIR:", ROOT_DIR)
prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)
SYSTEM_RULES = prompt_config["rag_assistant_prompt"]["role"]
PERSONA = prompt_config["rag_assistant_prompt"]["PERSONA"]
# SYSTEM_RULES = config.get("instruction")
# PERSONA =  config.get("role")


print("SYSTEM_RULES:", PERSONA)



# {'description': 'ASK-THE-documents AI assistant system prompt for chromadb', 
# 'role': 'You are an AI assistant for answering questions about the informaiton stored in the chromdb. 
# Below books are stored in the chromadb:\n1. "Clouding Computing -  100 Key Questions & Answers"\n2. 
# "Empowering Cybersecurity"\n3. "Generative AI -  Revolution"\n4. "Mastering Data Governance"\n\nYou are given related
# conntents to the topic and question will be asked to you.you will provide the answer briefly in approx 5 lines..\n',
#  'style_or_tone': ['Use clear, concise language with bullet points where appropriate.'], 'instruction': 
#  "Given the some documents that should be relevant to the user's question, answer the user's question.", 
#  'output_constraints': ['If you don\'t know the answer, just say "Hmm, I\'m not sure." Don\'t try to make up an answer.', 
#  'If the question is not about the questions or the details stored in the data store , 
#  politely inform them that you are tuned to only answer questions about money spent on restaurants .'], 
#  'output_format': ['Provide answers in markdown format.', 'Provide concise answers in bullet points when relevant.']}