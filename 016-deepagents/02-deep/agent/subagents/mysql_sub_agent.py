
from agent.config_load import config_dict
from tools.mysql_tools import select_tables,select_table_data,execute_custom_select_sql

mysql_sub_agent = {
    "name": config_dict['sub_agents']['db']['name'],
    "description":config_dict['sub_agents']['db']['description'],
    "system_prompt":config_dict['sub_agents']['db']['system_prompt'],
    "tools":[select_tables,select_table_data,execute_custom_select_sql]
}