import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def evaluate_grader(model, tokenizer, val_rows: list[dict]) -> dict:
    model.eval()
    valid_json, correct = 0, 0
    for row in val_rows:
        prompt = tokenizer.apply_chat_template(
            row["messages"][:2], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = int(inputs["input_ids"].size(1))
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out.squeeze(0)[prompt_len:]
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        expected = json.loads(row["messages"][-1]["content"])["is_sufficient"]
        try:
            parsed = json.loads(reply[reply.find("{") : reply.rfind("}") + 1])
            valid_json += 1
            if bool(parsed.get("is_sufficient")) == bool(expected):
                correct += 1
        except Exception:
            pass
    n = len(val_rows)
    return {
        "total": n,
        "json_valid_rate": round(valid_json / n, 4),
        "accuracy": round(correct / n, 4),
    }


def main() -> None:
    val_rows = [
        json.loads(line)
        for line in Path("fine_tuning/val.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_cfg, dtype=torch.float16, device_map="auto"
    )
    base_model.config.torch_dtype = torch.float16
    base_metrics = evaluate_grader(base_model, tokenizer, val_rows)

    model = prepare_model_for_kbit_training(base_model)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    ds = load_dataset(
        "json",
        data_files={
            "train": "fine_tuning/train.jsonl",
            "validation": "fine_tuning/val.jsonl",
        },
    )
    sft_cfg = SFTConfig(
        output_dir="./sysarch-grader-qlora",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        fp16=False,
        bf16=False,
        logging_steps=10,
        save_strategy="no",
        max_length=1024,
    )
    trainer = SFTTrainer(
        model=model, args=sft_cfg, train_dataset=ds["train"], processing_class=tokenizer
    )
    trainer.train()

    ft_metrics = evaluate_grader(model, tokenizer, val_rows)
    model.save_pretrained("./sysarch-grader-qlora/adapter")
    Path("fine_tuning/benchmark_results.json").write_text(
        json.dumps(
            {"model": MODEL_ID, "base": base_metrics, "qlora": ft_metrics}, indent=2
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
