import json

SYSTEM_PROMPT = """
你是中文智能助手的自然语言理解模块。你的任务是根据用户文本识别一个意图，并抽取零个或多个槽位。

候选意图：
alarm_query、alarm_remove、alarm_set、audio_volume_down、audio_volume_mute、audio_volume_other、audio_volume_up、calendar_query、calendar_remove、calendar_set、cooking_query、cooking_recipe、datetime_convert、datetime_query、email_addcontact、email_query、email_querycontact、email_sendemail、general_greet、general_joke、general_quirky、iot_cleaning、iot_coffee、iot_hue_lightchange、iot_hue_lightdim、iot_hue_lightoff、iot_hue_lighton、iot_hue_lightup、iot_wemo_off、iot_wemo_on、lists_createoradd、lists_query、lists_remove、music_dislikeness、music_likeness、music_query、music_settings、news_query、play_audiobook、play_game、play_music、play_podcasts、play_radio、qa_currency、qa_definition、qa_factoid、qa_maths、qa_stock、recommendation_events、recommendation_locations、recommendation_movies、social_post、social_query、takeaway_order、takeaway_query、transport_query、transport_taxi、transport_ticket、transport_traffic、weather_query

候选槽位：
alarm_type、app_name、artist_name、audiobook_author、audiobook_name、business_name、business_type、change_amount、coffee_type、color_type、cooking_type、currency_name、date、definition_word、device_type、drink_type、email_address、email_folder、event_name、food_type、game_name、game_type、general_frequency、house_place、ingredient、joke_type、list_name、meal_type、media_type、movie_name、movie_type、music_album、music_descriptor、music_genre、news_topic、order_type、person、personal_info、place_name、player_setting、playlist_name、podcast_descriptor、podcast_name、radio_name、relation、song_name、sport_type、time、time_zone、timeofday、transport_agency、transport_descriptor、transport_name、transport_type、weather_descriptor

请严格遵守以下规则：
1. 只输出一行JSON，不要解释，不要输出Markdown代码块。
2. JSON顶层必须且只能包含intent和slots两个字段。
3. intent必须是候选意图之一。
4. slots必须是数组；每个元素必须且只能包含name和value两个字符串字段。
5. name必须是候选槽位之一。
6. value必须从用户文本中逐字复制，不要改写、补充或翻译。
7. 槽位按照它们在用户文本中的出现顺序排列。
8. 同名槽位可以重复出现，不要合并。
9. 没有槽位时输出空数组。

输出格式：
{"intent":"<候选意图之一>","slots":[{"name":"<候选槽位之一>","value":"<用户原文片段>"}]}
"""

def to_message(exapmle, include_assistant=False):
    messages = []

    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }

    user_message = {
        "role": "user",
        "content": f"用户文本:{exapmle['text']}"
    }

    messages.append(system_message)
    messages.append(user_message)

    if include_assistant:
        
        assistant_context = json.dumps(
            {'intent': exapmle['intent'], 'slots': exapmle['slots']},
            ensure_ascii=False  # 中文非ascii
        )

        assistant_message = {
            "role": "assistant",
            "content": assistant_context
        }
        messages.append(assistant_message)


    return {"messages": messages}
