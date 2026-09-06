from tools.tavily_tool import internet_search
from agent.config_load import config_dict

tavily_sub_agent = {
    "name": config_dict['sub_agents']['tavily']['name'],
    "description":config_dict['sub_agents']['tavily']['description'],
    "system_prompt":config_dict['sub_agents']['tavily']['system_prompt'],
    "tools":[internet_search]
}