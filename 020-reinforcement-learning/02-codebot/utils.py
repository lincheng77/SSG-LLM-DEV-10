import torch.cuda
import torch.nn.functional as F

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=1024,
    temperature=1.0,
):
    model.eval()

    # 1. 将提示词转换成token ids
    device = next(model.parameters()).device # 获取模型参数所在的设备
    ids = tokenizer.encode(prompt)
    # [ids] 表示添加批次维度: (T,) --> (1, T)
    ids = torch.tensor([ids], device=device, dtype=torch.long)

    # 2. 保存已经生成的 token 变量
    # ids模型输入的 滑动窗口                    超过 256 会被砍头           喂给`model(ids)`
    # generated_ids 生成内容的 完整档案         永远不砍                   最后`decode` 出完整文本
    generated_ids = ids.clone() 

    # 3. 生成token循环
    for _ in range(max_new_tokens):
        # 超过上下文长度，只使用末尾256个token
        if ids.size(1) > model.max_context_len:
            ids = ids[:, -model.max_context_len:]
        
        # 预测下一个token
        logits = model(ids)[:, -1, :]
        if temperature == 0:
            # 贪婪策略
            next_id = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        # 如果next_id 是eos token， 则生成结束
        if next_id.item() == tokenizer.eot_token:
            break

        # 将新生成的token追加到上下文的末尾
        ids = torch.cat((ids, next_id), dim=1)
        generated_ids = torch.cat((generated_ids, next_id), dim=1)

    # 4. 解码成文本然后返回
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text
    