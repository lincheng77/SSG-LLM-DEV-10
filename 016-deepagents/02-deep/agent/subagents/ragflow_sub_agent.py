
from agent.config_load import config_dict
from tools.ragflow_tools import search_assistant_list, ask_ragflow_assistant

ragflow_sub_agent = {
    "name": config_dict['sub_agents']['ragflow']['name'],
    "description":config_dict['sub_agents']['ragflow']['description'],
    "system_prompt":config_dict['sub_agents']['ragflow']['system_prompt'],
    "tools":[search_assistant_list, ask_ragflow_assistant]
}