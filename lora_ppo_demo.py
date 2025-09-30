import json
import os
import sys
import time
import uuid
import re
from typing import Dict, Any

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

from config import cfg


def ensure_dirs():
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)


def get_dtype():
    if cfg.compute_dtype == "bfloat16":
        return torch.bfloat16
    if cfg.compute_dtype == "float16":
        return torch.float16
    return torch.float32


def _local_model_path(repo_id: str) -> str:
    # Create a filesystem-friendly subfolder name (replace slashes)
    safe = repo_id.replace("/", "__")
    return os.path.join(cfg.models_cache_dir, safe)


def load_models_and_tokenizer():
    dtype = get_dtype()

    quant = None
    if cfg.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=dtype,
        )

    # Prefer local cached path if exists, else fall back to repo id
    local_path = _local_model_path(cfg.model_name)
    load_id = local_path if os.path.isdir(local_path) else cfg.model_name
    if os.path.isdir(local_path):
        print(f"[Cache] Using local model cache: {local_path}")
    else:
        print(f"[Cache] No local cache found. Will download from HF: {cfg.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(load_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        load_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        quantization_config=quant,
        device_map="auto",
    )

    # If we loaded from HF (not local), save a copy into local cache BEFORE applying LoRA
    if not os.path.isdir(local_path):
        try:
            os.makedirs(local_path, exist_ok=True)
            print(f"[Cache] Saving model/tokenizer locally to: {local_path}")
            tokenizer.save_pretrained(local_path)
            # Save the base model attached to the value-head wrapper
            policy.pretrained_model.save_pretrained(local_path)
        except Exception as e:
            print(f"[Cache] Skipped saving local cache due to: {e}")

    # Apply LoRA to the policy base model (value head is separate)
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.lora_target_modules),
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy_pretrained = policy.pretrained_model
    if hasattr(policy_pretrained, "peft_config"):
        print("[Init] Detected existing PEFT adapters; skipping duplicate LoRA injection.")
    else:
        policy_pretrained = get_peft_model(policy_pretrained, lora_cfg)
        policy.pretrained_model = policy_pretrained

    # Reference model for PPO KL (keep on CPU to save VRAM). Use plain CausalLM (no value head).
    ref_model = AutoModelForCausalLM.from_pretrained(
        load_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
    )

    return policy, ref_model, tokenizer


def format_chat_prompt(question: str) -> str:
    # Avoid role prefixes to prevent the model from continuing a chat transcript
    return (
        "You are a helpful tutor. Answer the question clearly in 1-3 short paragraphs. "
        "Do not include any role prefixes like 'User:' or 'Assistant:'.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def format_quiz_prompt(question: str) -> str:
    # Important: quiz generated from Q only (prevents reward hacking via A leakage)
    # Ask for strict JSON schema with correct_option included but we will not reveal it to user.
    return (
        "Create a single multiple-choice quiz derived only from the user's question. "
        "Do not use any prior model responses. Respond with strict JSON ONLY in this schema: \n"
        '{"quiz_id": "<uuid>", "quiz_question": "<text>", '
        '"options": ["A: ...", "B: ...", "C: ...", "D: ..."], "correct_option": "A|B|C|D"}'
        f"\nUser question: {question}\nJSON:"
    )


def _strict_quiz_prompt(question: str) -> str:
    return (
        "Create ONE multiple-choice quiz strictly from the user's question only. "
        "Return ONLY valid JSON with EXACTLY these fields and nothing else: "
        '{"quiz_id": "<uuid>", "quiz_question": "<text>", '
        '"options": ["A: ...", "B: ...", "C: ...", "D: ..."], "correct_option": "A|B|C|D"}'
        " The value of correct_option MUST be a single uppercase letter A, B, C, or D. "
        f"User question: {question}\nJSON:"
    )


@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_prompt_tokens)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    out = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return out[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)) :].strip()


def normalize_correct_option(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            idx = int(value)
            if 0 <= idx <= 3:
                return "ABCD"[idx]
        except Exception:
            pass
    if isinstance(value, str):
        up = value.strip().upper()
        if up in ["A", "B", "C", "D"]:
            return up
        m = re.search(r"\b([ABCD])\b", up)
        if m:
            return m.group(1)
        m2 = re.search(r"\b([1-4])\b", up)
        if m2:
            return "ABCD"[int(m2.group(1)) - 1]
    return None


def parse_quiz(json_str: str) -> Dict[str, Any]:
    # Try to extract the first JSON object in the output
    start = json_str.find("{")
    end = json_str.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Quiz JSON not found")
    obj = json.loads(json_str[start : end + 1])
    # Basic validation
    assert "quiz_id" in obj and "quiz_question" in obj and "options" in obj and "correct_option" in obj
    # Ensure 4 options
    if not isinstance(obj["options"], list) or len(obj["options"]) != 4:
        raise ValueError("Quiz must contain exactly 4 options")
    # Normalize correct_option
    norm = normalize_correct_option(obj["correct_option"]) 
    if norm is None:
        raise ValueError("correct_option is invalid")
    obj["correct_option"] = norm
    return obj


def clean_answer(text: str) -> str:
    # Remove stray role prefixes if the model emits chat-like transcripts
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("User:") or s.startswith("Assistant:"):
            continue
        lines.append(line)
    out = "\n".join(lines).strip()
    return out


def main():
    ensure_dirs()
    logger.add(os.path.join(cfg.log_dir, "run.log"))
    logger.info("Loading models and tokenizer...")
    print(f"[Init] Model: {cfg.model_name}")
    print(f"[Init] 4-bit quantization: {cfg.load_in_4bit} (type={cfg.bnb_4bit_quant_type})")
    policy, ref_model, tokenizer = load_models_and_tokenizer()
    print("[Init] Models loaded. Policy on device:", policy.pretrained_model.device)
    print("[Init] Reference model on CPU for KL computation.")

    # PPO trainer (robust across TRL versions)
    ppo_config = PPOConfig()
    # Set known fields if they exist in this TRL version
    if hasattr(ppo_config, "learning_rate"):
        ppo_config.learning_rate = cfg.learning_rate
    if hasattr(ppo_config, "batch_size"):
        ppo_config.batch_size = cfg.batch_size
    if hasattr(ppo_config, "mini_batch_size"):
        ppo_config.mini_batch_size = 1
    if hasattr(ppo_config, "gradient_accumulation_steps"):
        ppo_config.gradient_accumulation_steps = 1
    # ppo epochs can be named differently across versions
    if hasattr(ppo_config, "ppo_epochs"):
        ppo_config.ppo_epochs = cfg.ppo_epochs
    elif hasattr(ppo_config, "num_ppo_epochs"):
        ppo_config.num_ppo_epochs = cfg.ppo_epochs
    # target KL / cliprange naming can differ
    if hasattr(ppo_config, "target_kl"):
        ppo_config.target_kl = cfg.target_kl
    elif hasattr(ppo_config, "kl_target"):
        ppo_config.kl_target = cfg.target_kl
    if hasattr(ppo_config, "cliprange"):
        ppo_config.cliprange = cfg.cliprange
    elif hasattr(ppo_config, "cliprange_value"):
        ppo_config.cliprange_value = cfg.cliprange
    if hasattr(ppo_config, "log_with"):
        ppo_config.log_with = None

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    logger.info("Tutor loop ready. Type Ctrl+C to exit.")
    print("[Ready] Tutor loop started. Press Ctrl+C to exit.")
    step = 0
    quiz_store: Dict[str, str] = {}  # quiz_id -> correct_option (kept off-model)

    try:
        while True:
            q = input("\nEnter your question (Q): ").strip()
            if not q:
                continue

            # 1) Generate answer A
            prompt = format_chat_prompt(q)
            print("\n[Step] Generating answer...")
            t0 = time.time()
            a = generate_text(
                ppo_trainer.model.pretrained_model,
                tokenizer,
                prompt,
                temperature=cfg.answer_temperature,
                top_p=cfg.top_p,
                max_new_tokens=cfg.max_new_tokens,
            )
            print(f"[Done] Answer generated in {time.time()-t0:.2f}s\n")
            # Optionally show a preview
            a = clean_answer(a)
            print("[Answer]\n" + a[:] + ("..." if len(a) > 500 else ""))

            # 2) Generate quiz from Q only (no A)
            quiz_prompt = format_quiz_prompt(q)
            print("\n[Step] Generating quiz (from Q only)...")
            t1 = time.time()
            quiz_raw = generate_text(
                ppo_trainer.model.pretrained_model,
                tokenizer,
                quiz_prompt,
                temperature=cfg.quiz_temperature,
                top_p=cfg.top_p,
                max_new_tokens=256,
            )
            print(f"[Done] Quiz draft generated in {time.time()-t1:.2f}s")
            # Parse quiz JSON and store correct_option separately
            try:
                print("quiz_raw", quiz_raw)
                quiz = parse_quiz(quiz_raw)

            except Exception as e:
                print(f"[Warn] Quiz parse failed: {e}. Retrying with stricter prompt...")
                strict_prompt = _strict_quiz_prompt(q)
                quiz_raw2 = generate_text(
                    ppo_trainer.model.pretrained_model,
                    tokenizer,
                    strict_prompt,
                    temperature=cfg.quiz_temperature,
                    top_p=cfg.top_p,
                    max_new_tokens=256,
                )
                try:
                    quiz = parse_quiz(quiz_raw2)
                except Exception as e2:
                    print(f"[Error] Second quiz parse failed: {e2}. Skipping this round.")
                    continue

            quiz_id = quiz.get("quiz_id") or str(uuid.uuid4())
            correct_option = quiz["correct_option"].strip()
            quiz_store[quiz_id] = correct_option  # never shown, never fed back

            # Show quiz (without correct_option)
            print("\n--- Quiz ---")
            print(quiz["quiz_question"])
            for opt in quiz["options"]:
                print(opt)
            user_choice = input("Your choice (A/B/C/D): ").strip().upper()
            if user_choice not in ["A", "B", "C", "D"]:
                user_choice = "A"

            reward = 1.0 if user_choice == correct_option else -1.0
            print(f"[Feedback] User chose {user_choice}. Correct was hidden. Reward: {reward:+.0f}")

            # PPO step uses the original (Q, A) pair and scalar reward
            # Prepare tensors
            print("[Step] PPO update...")
            q_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_prompt_tokens).input_ids.to(ppo_trainer.accelerator.device)
            a_ids = tokenizer(a, return_tensors="pt", truncation=True, max_length=cfg.max_new_tokens).input_ids.to(ppo_trainer.accelerator.device)

            # TRL expects rewards as tensors (not Python floats)
            score = torch.tensor(reward, dtype=torch.float32, device=ppo_trainer.accelerator.device)
            stats = ppo_trainer.step([q_ids[0]], [a_ids[0]], [score])
            # Summarize core PPO metrics if available
            kl = stats.get("ppo/kl", None)
            pl = stats.get("ppo/policy_loss", None)
            vl = stats.get("ppo/value_loss", None)
            print(f"[Done] PPO step complete. KL={kl}, policy_loss={pl}, value_loss={vl}")

            # Logging
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info({
                "ts": now,
                "Q": q,
                "A": a,
                "quiz_id": quiz_id,
                "user_answer": user_choice,
                "reward": reward,
                "ppo_stats": stats,
            })

            step += 1
            if step % cfg.save_every_n_steps == 0:
                # Save only LoRA adapters
                save_path = os.path.join(cfg.output_dir, f"step_{step}")
                os.makedirs(save_path, exist_ok=True)
                ppo_trainer.model.pretrained_model.save_pretrained(save_path)
                logger.info(f"Saved LoRA adapters to {save_path}")
                print(f"[Checkpoint] Saved LoRA adapters: {save_path}")

    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()


