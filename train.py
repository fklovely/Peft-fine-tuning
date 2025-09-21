from modelscope.msdatasets import MsDataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainerCallback
# ==================== 新增部分：自定义 Callback ====================
# 创建一个自定义的回调类，用于在每个 epoch 结束时打印信息
class EpochEndCallback(TrainerCallback):
    """
    一个在每个训练回合（Epoch）结束时触发的回调。
    """

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        在 on_epoch_end 事件中，从 state.log_history 获取并打印最新的训练日志。
        """
        # state.log_history 包含了所有步骤的日志记录
        # 我们找到最近一次记录的训练日志
        latest_log = None
        for log in reversed(state.log_history):
            if 'loss' in log:  # 训练日志通常包含 'loss' 键
                latest_log = log
                break

        if latest_log:
            epoch = latest_log.get('epoch', 'N/A')
            loss = latest_log.get('loss', 'N/A')
            learning_rate = latest_log.get('learning_rate', 'N/A')

            # 打印格式化的输出
            print(f"\n++++++++++ Epoch {epoch:.2f} 结束 ++++++++++")
            print(f"  最新训练 Loss: {loss}")
            print(f"  当前学习率: {learning_rate}")
            print(f"++++++++++++++++++++++++++++++++++++++\n")


# =================================================================

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoModel
)

# 1. 加载数据
ds = MsDataset.load('ethanfly/shose', subset_name='default', split='train')
local_model_path = "/root/autodl-tmp/Qwen1.5-1.8B-Chat"
# 2. 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(
    #'Qwen/Qwen1.5-1.8B-Chat',
    #trust_remote_code=True
    local_model_path,
    local_files_only=True
)

#model = AutoModelForCausalLM.from_pretrained(
#    'Qwen/Qwen1.5-1.8B-Chat',
#   trust_remote_code=True
#)
model = AutoModelForCausalLM.from_pretrained(local_model_path, local_files_only=True)
# 3. 配置 LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj","v_proj"],
)

model = get_peft_model(model, peft_config)
#model.print_trainable_parameters()

# 4. 数据预处理
def format_example(example):
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    if inp:
        prompt = f"用户: {instruction}\n{inp}\n助手:"
    else:
        prompt = f"用户: {instruction}\n助手:"
    full_text = prompt + " " + output
    return {"text": full_text}

ds = ds.map(format_example)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )

tokenized_ds = ds.map(tokenize)

# 5. DataCollator (动态 mask 语言模型)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 6. 训练参数
training_args = TrainingArguments(
    output_dir="output",
    learning_rate=5e-4,
    per_device_train_batch_size=2,   # 建议调小，1.8B模型占显存很大
    fp16=True,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
    gradient_accumulation_steps=8,
    logging_steps=10,
    save_total_limit=2,
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EpochEndCallback()]
)

# 8. 开始训练
trainer.train()



