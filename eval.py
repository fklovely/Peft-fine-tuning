import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------- 加载模型 --------------------
base_model_path = "/root/autodl-tmp/Qwen1.5-1.8B-Chat"
lora_model_path = "output/checkpoint-1280"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True
)
model = PeftModel.from_pretrained(model, lora_model_path)
model.eval()


# -------------------- Alpaca 风格多轮对话 --------------------
class AlpacaChatSession:
    def __init__(self, max_history=5):
        """
        max_history: 最多保留的历史轮数
        history: list of tuples [(instruction, input, assistant), ...]
        """
        self.max_history = max_history
        self.history = []

    def add_user_input(self, instruction, inp=""):
        self.history.append((instruction, inp, None))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def add_assistant_response(self, assistant_text):
        if self.history and self.history[-1][2] is None:
            instruction, inp, _ = self.history[-1]
            self.history[-1] = (instruction, inp, assistant_text)

    def build_prompt(self):
        """
        拼接历史对话，生成 prompt
        """
        prompt = ""
        for instruction, inp, assistant in self.history:
            if inp:
                user_text = f"用户: {instruction}\n{inp}\n助手:"
            else:
                user_text = f"用户: {instruction}\n助手:"
            prompt += user_text + ("\n" + assistant + "\n" if assistant else "\n")
        prompt += "助手:"  # 本轮回答起始
        return prompt


# -------------------- 生成函数 --------------------
def chat(session, model, tokenizer, max_new_tokens=200, temperature=0.7, top_p=0.9):
    prompt = session.build_prompt()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_text = response.split("助手:")[-1].strip()
    session.add_assistant_response(assistant_text)
    return assistant_text


# -------------------- 命令行交互 --------------------
def main():
    session = AlpacaChatSession(max_history=5)
    print("=== Alpaca 风格多轮对话 ===")
    print("输入 instruction 和 input（input 可为空），输入 exit 或 退出 结束对话\n")

    while True:
        instruction = input("Instruction: ").strip()
        if instruction.lower() in ["exit", "quit", "退出"]:
            break
        inp = input("Input (可为空): ").strip()
        session.add_user_input(instruction, inp)
        assistant_output = chat(session, model, tokenizer)
        print("助手:", assistant_output)
        print("-" * 50)


if __name__ == "__main__":
    main()