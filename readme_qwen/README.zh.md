[English](README.md)

# Qwen2-Boundless

## 简介

Qwen2-Boundless 是一个基于 Qwen2-1.5B-Instruct 微调的模型，专为回答各种类型的问题而设计，无论是道德的、违法的、色情的、暴力的内容，均可自由询问。该模型经过特殊的数据集训练，能够应对复杂和多样的场景。需要注意的是，微调数据集全部为中文，因此模型在处理中文时表现更佳。

> **警告**：本模型仅用于研究和测试目的，用户应遵循当地法律法规，并对自己的行为负责。

## 模型使用

你可以通过以下代码加载并使用该模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

device = "cuda" # the device to load the model onto
current_directory = os.path.dirname(os.path.abspath(__file__))

model = AutoModelForCausalLM.from_pretrained(
    current_directory,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(current_directory)

prompt = "Hello?"
messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### 连续对话

要实现连续对话，可以使用以下代码：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

device = "cuda"  # the device to load the model onto

# 获取当前脚本所在的目录
current_directory = os.path.dirname(os.path.abspath(__file__))

model = AutoModelForCausalLM.from_pretrained(
    current_directory,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(current_directory)

messages = [
    {"role": "system", "content": ""}
]

while True:
    # 获取用户输入
    user_input = input("User: ")

    # 将用户输入添加到对话中
    messages.append({"role": "user", "content": user_input})

    # 准备输入文本
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 生成响应
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解码并打印响应
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Assistant: {response}")

    # 将生成的响应添加到对话中
    messages.append({"role": "assistant", "content": response})
```

### 流式响应

对于需要流式响应的应用，使用以下代码：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.trainer_utils import set_seed
from threading import Thread
import random
import os

DEFAULT_CKPT_PATH = os.path.dirname(os.path.abspath(__file__))

def _load_model_tokenizer(checkpoint_path, cpu_only):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, resume_download=True)

    device_map = "cpu" if cpu_only else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype="auto",
        device_map=device_map,
        resume_download=True,
    ).eval()
    model.generation_config.max_new_tokens = 512    # For chat.

    return model, tokenizer

def _get_input() -> str:
    while True:
        try:
            message = input('User: ').strip()
        except UnicodeDecodeError:
            print('[ERROR] Encoding error in input')
            continue
        except KeyboardInterrupt:
            exit(1)
        if message:
            return message
        print('[ERROR] Query is empty')

def _chat_stream(model, tokenizer, query, history):
    conversation = [
        {'role': 'system', 'content': ''},
    ]
    for query_h, response_h in history:
        conversation.append({'role': 'user', 'content': query_h})
        conversation.append({'role': 'assistant', 'content': response_h})
    conversation.append({'role': 'user', 'content': query})
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors='pt',
    )
    inputs = inputs.to(model.device)
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=inputs,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

def main():
    checkpoint_path = DEFAULT_CKPT_PATH
    seed = random.randint(0, 2**32 - 1)  # 随机生成一个种子
    set_seed(seed)  # 设置随机种子
    cpu_only = False

    history = []

    model, tokenizer = _load_model_tokenizer(checkpoint_path, cpu_only)

    while True:
        query = _get_input()

        print(f"\nUser: {query}")
        print(f"\nAssistant: ", end="")
        try:
            partial_text = ''
            for new_text in _chat_stream(model, tokenizer, query, history):
                print(new_text, end='', flush=True)
                partial_text += new_text
            print()
            history.append((query, partial_text))

        except KeyboardInterrupt:
            print('Generation interrupted')
            continue

if __name__ == "__main__":
    main()
```

## 数据集

Qwen2-Boundless 模型使用了特殊的 `bad_data.json` 数据集进行微调，该数据集包含了广泛的文本内容，涵盖道德、法律、色情及暴力等主题。由于微调数据集全部为中文，因此模型在处理中文时表现更佳。如果你有兴趣了解或使用该数据集，可以通过以下链接获取：

- [bad_data.json 数据集](https://huggingface.co/datasets/ystemsrx/bad_data.json)

同时我们也从 [这个文件](https://github.com/Clouditera/SecGPT/blob/main/secgpt-mini/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9B%9E%E7%AD%94%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98-cot.txt) 中整理、清洗出一部分与网络安全相关的数据进行训练。

## GitHub 仓库

更多关于该模型的细节以及持续更新，请访问我们的 GitHub 仓库：

- [GitHub: ystemsrx/Qwen2-Boundless](https://github.com/ystemsrx/Qwen2-Boundless)

## 声明

本模型提供的所有内容仅用于研究和测试目的，模型开发者不对任何可能的滥用行为负责。使用者应遵循相关法律法规，并承担因使用本模型而产生的所有责任。