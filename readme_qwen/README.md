---
license: apache-2.0
datasets:
- ystemsrx/Bad_Data_Alpaca
language:
- zh
library_name: transformers
pipeline_tag: text2text-generation
tags:
- Qwen
- causal-lm
- fine-tuned
- ethics
- Chinese
---

[中文](README.zh.md)

# Qwen2-Boundless

## Overview

Qwen2-Boundless is a fine-tuned model based on Qwen2-1.5B-Instruct, designed to answer various types of questions, including those involving ethical, illegal, pornographic, and violent content. This model has been specifically trained on a dataset that allows it to handle complex and diverse scenarios. It is important to note that the fine-tuning dataset is entirely in Chinese, so the model performs better in Chinese.

> **Warning**: This model is intended for research and testing purposes only. Users should comply with local laws and regulations and are responsible for their actions.

## How to Use

You can load and use the model with the following code:

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

### Continuous Conversation

To enable continuous conversation, use the following code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

device = "cuda"  # the device to load the model onto

# Get the current script's directory
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
    # Get user input
    user_input = input("User: ")

    # Add user input to the conversation
    messages.append({"role": "user", "content": user_input})

    # Prepare the input text
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # Generate a response
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode and print the response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Assistant: {response}")

    # Add the generated response to the conversation
    messages.append({"role": "assistant", "content": response})
```

### Streaming Response

For applications requiring streaming responses, use the following code:

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
    seed = random.randint(0, 2**32 - 1)  # Generate a random seed
    set_seed(seed)  # Set the random seed
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

## Dataset

The Qwen2-Boundless model was fine-tuned using a specific dataset named `bad_data.json`, which includes a wide range of text content covering topics related to ethics, law, pornography, and violence. The fine-tuning dataset is entirely in Chinese, so the model performs better in Chinese. If you are interested in exploring or using this dataset, you can find it via the following link:

- [bad_data.json Dataset](https://huggingface.co/datasets/ystemsrx/Bad_Data_Alpaca)

And also we used some cybersecurity-related data that was cleaned and organized from [this file](https://github.com/Clouditera/SecGPT/blob/main/secgpt-mini/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9B%9E%E7%AD%94%E9%9D%A2%E9%97%AE%E9%A2%98-cot.txt).

## GitHub Repository

For more details about the model and ongoing updates, please visit our GitHub repository:

- [GitHub: ystemsrx/Qwen2-Boundless](https://github.com/ystemsrx/Qwen2-Boundless)

## License

This model and dataset are open-sourced under the Apache 2.0 License.

## Disclaimer

All content provided by this model is for research and testing purposes only. The developers of this model are not responsible for any potential misuse. Users should comply with relevant laws and regulations and are solely responsible for their actions.