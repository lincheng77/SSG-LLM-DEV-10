import os

import torch

# 指定 Hugging Face 镜像源（本机访问 huggingface.co 不通，
# 不设置的话 transformers 装 peft 后会联网检查 adapter 导致加载卡住）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 禁用 Xet（大文件的默认下载方式）：Xet 连的不是镜像，会报 401 下不下来，禁用后走普通下载就能从镜像下
os.environ['HF_HUB_DISABLE_XET'] = '1'

from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from message import to_message
import json


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'


def evaluate(model, tokenizer):
    # 加载数据
    dataset = load_dataset('json', data_files=str(DATA_DIR / 'test.jsonl'))['train']
    dataset = dataset.map(
        to_message, batched=False, remove_columns=['text'],
        fn_kwargs={'include_assistant': False}  # 评测时不能把标准答案放进 prompt，否则模型只会在答案后续写，准确率必为 0；intent/slots 列仍保留用于比对
    )
    print(dataset[:1])
    # 评估
    #{'intent': ['alarm_set'], 'slots': [[{'name': 'date', 'value': '这周'}, {'name': 'time', 'value': '五点'}]], 'messages': [[{'role': 'system', 'content': '\n你是中文智能助手的自然语言理解模块。你的任务是根据用户文本识别一个意图，并抽取零个或多个槽位。\n\n候选意图：\nalarm_query、alarm_remove、alarm_set、audio_volume_down、audio_volume_mute、audio_volume_other、audio_volume_up、calendar_query、calendar_remove、calendar_set、cooking_query、cooking_recipe、datetime_convert、datetime_query、email_addcontact、email_query、email_querycontact、email_sendemail、general_greet、general_joke、general_quirky、iot_cleaning、iot_coffee、iot_hue_lightchange、iot_hue_lightdim、iot_hue_lightoff、iot_hue_lighton、iot_hue_lightup、iot_wemo_off、iot_wemo_on、lists_createoradd、lists_query、lists_remove、music_dislikeness、music_likeness、music_query、music_settings、news_query、play_audiobook、play_game、play_music、play_podcasts、play_radio、qa_currency、qa_definition、qa_factoid、qa_maths、qa_stock、recommendation_events、recommendation_locations、recommendation_movies、social_post、social_query、takeaway_order、takeaway_query、transport_query、transport_taxi、transport_ticket、transport_traffic、weather_query\n\n候选槽位：\nalarm_type、app_name、artist_name、audiobook_author、audiobook_name、business_name、business_type、change_amount、coffee_type、color_type、cooking_type、currency_name、date、definition_word、device_type、drink_type、email_address、email_folder、event_name、food_type、game_name、game_type、general_frequency、house_place、ingredient、joke_type、list_name、meal_type、media_type、movie_name、movie_type、music_album、music_descriptor、music_genre、news_topic、order_type、person、personal_info、place_name、player_setting、playlist_name、podcast_descriptor、podcast_name、radio_name、relation、song_name、sport_type、time、time_zone、timeofday、transport_agency、transport_descriptor、transport_name、transport_type、weather_descriptor\n\n请严格遵守以下规则：\n1. 只输出一行JSON，不要解释，不要输出Markdown代码块。\n2. JSON顶层必须且只能包含intent和slots两个字段。\n3. intent必须是候选意图之一。\n4. slots必须是数组；每个元素必须且只能包含name和value两个字符串字段。\n5. name必须是候选槽位之一。\n6. value必须从用户文本中逐字复制，不要改写、补充或翻译。\n7. 槽位按照它们在用户文本中的出现顺序排列。\n8. 同名槽位可以重复出现，不要合并。\n9. 没有槽位时输出空数组。\n\n输出格式：\n{"intent":"<候选意图之一>","slots":[{"name":"<候选槽位之一>","value":"<用户原文片段>"}]}\n'}, {'role': 'user', 'content': '用户文本:这周五点叫我起床'}, {'role': 'assistant', 'content': '"这周五点叫我起床"'}]]}
    batch_size = 32
    correct = 0
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i: i + batch_size]

        # chat template
        text_list = tokenizer.apply_chat_template(batch['messages'],
                                                  tokenize=False,
                                                  add_generation_prompt=True,  # 添加生成提示
                                                  enable_thinking=False)       # 添加思考空白标签

        # tokenize
        inputs = tokenizer(text_list,
                           padding=True,              # 对不同长度的序列进行补齐，使同一 batch 中的输入长度一致
                           padding_side='left',       # 从序列左侧进行 padding，常用于自回归语言模型的批量推理
                           truncation=True,           # 当输入长度超过最大长度时，自动截断超出的部分
                           return_tensors='pt').to(model.device)

        # generate
        output_ids = model.generate(**inputs, do_sample=False, num_beams = 1, max_new_tokens = 2048)
        seq_len = inputs['input_ids'].shape[1]
        output_ids = output_ids[:, seq_len:]
        results = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for result, intent, slots in zip(results, batch['intent'], batch['slots']):
            try:
                result_dict = json.loads(result)
            except json.decoder.JSONDecodeError:
                result_dict = None

            expected = {'intent': intent, 'slots': slots}

            if result_dict == expected:
                correct += 1
    return correct / len(dataset)

            

if __name__ == '__main__':
    # # 基线测试(0.28)
    # model_name = "Qwen/Qwen3-0.6B"
    # tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name,
    #                                              torch_dtype='auto',
    #                                              device_map="auto",
    #                                              local_files_only=True)
    # print(evaluate(model, tokenizer))


    # # 微调模型测试(0.61)
    # model_name = str(BASE_DIR / 'output' / 'full' / 'best')
    # tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name,
    #                                              torch_dtype='auto',
    #                                              device_map="auto",
    #                                              local_files_only=True)
    # print(evaluate(model, tokenizer))


    # # lora模型测试(0.48)
    # model_name = str(BASE_DIR / 'output' / 'lora' / 'best')
    # tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    # # LoRA 保存的是 adapter，需用 AutoPeftModelForCausalLM 加载，不能用 AutoModelForCausalLM
    # model = AutoPeftModelForCausalLM.from_pretrained(model_name,
    #                                                  torch_dtype='auto',
    #                                                  device_map="auto",
    #                                                  local_files_only=True)
    # #model = model.merge_and_unload()  # 将 adapter 权重合并进基座，推理更快（可选）
    # print(evaluate(model, tokenizer))

    # qlora模型测试(0.37)
    # model_name = str(BASE_DIR / 'output' / 'qlora' / 'best')
    # tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    # # 量化
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16 # 指定 4-bit 量化权重参与计算时，实际使用什么浮点精度来做矩阵运算
    # )
    # model = AutoPeftModelForCausalLM.from_pretrained(model_name,
    #                                                  torch_dtype='auto',
    #                                                  quantization_config = bnb_config,
    #                                                  local_files_only=True)
    # #model = model.merge_and_unload()  # 将 adapter 权重合并进基座，推理更快（可选）
    # print(evaluate(model, tokenizer))


    # # 基线测试Qwen3-4B(0.11)
    # model_name = "Qwen/Qwen3-4B"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name,
    #                                              torch_dtype='auto',
    #                                              device_map="auto")
    # print(evaluate(model, tokenizer))


    # Qwen3-4B微调模型测试(0.65)
    model_name = str(BASE_DIR / 'output' / 'zero' / 'best')
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype='auto',
                                                 device_map="auto",
                                                 local_files_only=True)
    print(evaluate(model, tokenizer))