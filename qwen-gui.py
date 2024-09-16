from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import customtkinter

# 判断是否有GPU可用，若可用则使用GPU，否则使用CPU
device = "cuda" if torch.cuda.is_available() else "cpu" 
# 获取当前文件的路径
current_directory = os.path.dirname(os.path.abspath(__file__))  
# 从本地加载预训练的模型
model = AutoModelForCausalLM.from_pretrained(current_directory, torch_dtype="auto", device_map="auto").to(device)  
# 从本地加载分词器
tokenizer = AutoTokenizer.from_pretrained(current_directory)  
  
# 定义一个列表，用于存储聊天信息
messages = [  
    {"role": "system", "content": ""}  
] 

# 定义一个函数，用于生成回复
def generate_response(user_input):  
    global messages  
    # 将用户输入添加到聊天信息列表中
    messages.append({"role": "user", "content": user_input})  
      
    # 使用分词器将聊天信息转换为文本
    text = tokenizer.apply_chat_template(  
        messages,  
        tokenize=False,  
        add_generation_prompt=True  
    )  
    # 对文本进行分词
    model_inputs = tokenizer([text], return_tensors="pt").to(device)  
      
    # 使用模型生成回复
    generated_ids = model.generate(  
        model_inputs.input_ids,  
        max_new_tokens=512  
    )  
    # 获取生成的ids
    generated_ids = [  
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)  
    ]  
      
    # 将ids转换为文本
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]  
    # 将生成的回复添加到聊天信息列表中
    messages.append({"role": "assistant", "content": response})  
    # 返回生成的回复
    return response  

# 设置应用的 appearance mode 和 default color theme
customtkinter.set_appearance_mode("system")
customtkinter.set_default_color_theme("blue")
# 创建一个应用
app = customtkinter.CTk()
app.geometry("600x600")
app.title("通义千问2-无尽版")

# 创建一个用户协议窗口
def show_agreement():
    # 定义用户协议文本
    agreement_text = '''
    用户协议：

    Qwen2-Boundless是由 ystemsrx 从 Qwen2-1.5B-Instruct 模型微调而来的高级语言模型。
    
    它能够生成对各种问题的回答，包括那些传统商业模型可能会回避的内容。

    （如暴力、色情、违法行为和不道德行为）
    
    此外，该模型也能回答常规的合适问题，适用于多种应用场景。

    该模型在包含潜在敏感或争议内容的数据集上进行了微调，
    
    包括暴力、色情、违法行为和不道德行为。
    
    用户在使用该模型时应充分意识到这些内容，建议在受控环境下应用此模型。

    Qwen2-Boundless 的创建者不认可或支持任何非法或不道德的使用。
    
    该模型仅供研究用途，用户应确保其使用符合所有适用的法律和道德规范。
    '''
    # 创建一个用户协议窗口
    agreement_window = customtkinter.CTkToplevel()
    agreement_window.title('用户协议')
    agreement_window.geometry('600x400')
    # 在用户协议窗口中显示用户协议文本
    agreement_label = customtkinter.CTkLabel(agreement_window, text=agreement_text, font=('Arial', 12))
    agreement_label.pack(pady=20)
 
dialog = customtkinter.CTkInputDialog(text="请输入您的昵称:", title="用户注册")
username=dialog.get_input()
show_agreement()   

text_box = customtkinter.CTkTextbox(app, height=300, width=500)
text_box.pack(pady=20)

input_box = customtkinter.CTkEntry(app, height=30, width=500)
input_box.pack(pady=20)

send_button = customtkinter.CTkButton(app, text="发送", command=lambda: send_message(input_box.get()))
send_button.pack(pady=20)
def send_message(user_input):
    response = generate_response(user_input)
    text_box.insert("end", f"{username if username is not None else 'User'}: {user_input}\n")
    text_box.insert("end", f"通义千问2-无尽版: {response}\n")
    input_box.delete(0, "end")
    text_box.see("end")

app.mainloop()