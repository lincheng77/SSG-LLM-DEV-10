
import os

import torch

# 指定 Hugging Face 镜像源（本机访问 huggingface.co 不通，
# 不设置的话 transformers 装 peft 后会联网检查 adapter 导致加载卡住）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


from peft import LoraConfig, TaskType, get_peft_model
from pathlib import Path
from datasets import load_dataset, disable_caching
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from message import to_message
from trl import SFTConfig, SFTTrainer

# 全局禁用 datasets 缓存：指纹只含"message.to_message"引用不含代码（dill 按引用序列化导入函数），
# 修改 message.py 后指纹不变，SFTTrainer 内部 tokenize 的 map 会命中旧缓存（旧缓存曾导致模型学复述、准确率为 0）
disable_caching()


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output' / 'qlora'

def train():
    model_name = "Qwen/Qwen3-0.6B"

    # 量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 # 指定 4-bit 量化权重参与计算时，实际使用什么浮点精度来做矩阵运算
    )

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                quantization_config=bnb_config,
                                                local_files_only=True)

    # lora配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型（自回归生成，适配 Qwen3）
        r=8,                           # LoRA 低秩矩阵的秩，决定可训练参数量，8 为常用值
        lora_alpha=16,                 # LoRA 缩放系数，缩放比 = alpha / r = 2，控制微调强度
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,  # LoRA 分支的 dropout 比例，防止过拟合
        bias='none'         # 不训练偏置项，只训练低秩矩阵，进一步减少参数量
    )


    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)

    dataset_dict = load_dataset('json', data_files={
        'train': str(DATA_DIR / 'train.jsonl'),
        'validation': str(DATA_DIR / 'validation.jsonl')
    })

    dataset_dict = dataset_dict.map(to_message, batched=False, remove_columns=['text', 'intent', 'slots'],
                                    fn_kwargs={"include_assistant": True},
                                    load_from_cache_file=False)  # 禁用 map 缓存：防止命中修改 to_message 之前的旧缓存数据（旧缓存曾导致模型学会复述用户文本、评测准确率为0）


    # 配置参数
    args = SFTConfig(
        max_length=10240,
        assistant_only_loss=True,
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=8,
        num_train_epochs=1,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        bf16=True,
        bf16_full_eval=True,  # 评估阶段同样使用 bf16 混合精度，加快验证集评估并节省显存（精度损失可忽略）
        gradient_checkpointing=False,
        report_to="tensorboard",
        logging_strategy="steps",
        logging_steps=20,
        per_device_eval_batch_size=16,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=3,
        load_best_model_at_end=True,  # 训练结束时自动加载验证集上表现最好的模型权重（用于最终保存）
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # 指定的最优指标 eval_loss 越小越好，即选验证集 loss 最小的 checkpoint（配合metric_for_best_model使用）
    )


    # SFTTrainer
    trainer = SFTTrainer(
        model=peft_model,
        args=args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['validation'],
        processing_class=tokenizer
    )


    # 训练
    trainer.train()

    # 保存模型
    trainer.save_model(str(OUTPUT_DIR / 'best'))

if __name__ == '__main__':
    train()