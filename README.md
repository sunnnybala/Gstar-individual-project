# LoRA + PPO Tutor Demo (Minimal, No RAG)

This is a minimal, single-user dev loop for a tutor-style assistant trained with LoRA and PPO on a small instruct model (3–7B). It:

- Accepts a user question (Q) via CLI
- Generates an answer (A)
- Generates a multiple-choice quiz from Q only (JSON with hidden correct answer)
- Asks the user to pick an option
- Computes reward (+1/-1)
- Runs a PPO step to update only LoRA adapters (base model frozen) and the value head
- Periodically saves LoRA adapters to disk

No RAG, minimal dependencies, batch_size=1.

## Requirements

- Python 3.10+
- NVIDIA GPU recommended (works on laptop RTX 4080)
- CUDA-compatible PyTorch (installed via pip as pinned in requirements)

Install packages:

```bash
pip install -r requirements.txt
```

If bitsandbytes fails to load on native Windows, consider WSL2 Ubuntu or temporarily switch `load_in_4bit=False` and/or use a 3B model.

## Configure

Edit `config.py` to choose a model and hyperparameters:

- Default model: `Qwen/Qwen2-3B-Instruct` (4-bit QLoRA)
- To try 7B later: set `model_name = "mistralai/Mistral-7B-Instruct-v0.2"`
- Troubleshooting: set `load_in_4bit=False`

Key settings:
- LoRA targets: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- Generation: temp=0.2/0.3, top_p=0.9, max_new_tokens=256
- PPO: lr=1e-5, ppo_epochs=1, batch_size=1, target_kl=0.1

## Run

```bash
python lora_ppo_demo.py
```

You will be prompted for a question. The script will:
1) Generate an answer.
2) Generate a quiz JSON strictly from the question (never the answer), parse it, store `correct_option` off-model.
3) Show quiz (without `correct_option`) and ask for your choice.
4) Compute reward and perform a PPO step.
5) Save LoRA adapters every N steps to `./checkpoints/`.

Logs go to `./logs/run.log` and include: Q, A, quiz_id, user_answer, reward, PPO metrics, KL, timestamp.

### WSL2 setup (recommended on Windows)
1) Enable WSL and install Ubuntu:
   - In PowerShell (Admin):
     ```powershell
     wsl --install -d Ubuntu
     ```
   - Reboot if prompted, then launch Ubuntu and create a user.
2) In Ubuntu, install basics and CUDA-enabled PyTorch stack:
   ```bash
   sudo apt update && sudo apt install -y git python3-venv python3-pip
   python3 -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Note: Ensure your NVIDIA drivers are installed on Windows and WSL GPU is available (`nvidia-smi` inside WSL).
3) Run:
   ```bash
   python lora_ppo_demo.py
   ```

## Safety note (anti-reward-hacking)
- The quiz is generated only from Q, never from A.
- The `correct_option` is parsed from the JSON and kept only in memory.
- We never feed `correct_option` back into prompts or PPO; user never sees it.

## Expected VRAM
- 7B with 4-bit QLoRA + bs=1 should fit in ~5–7 GB.
- If OOM:
  - Reduce `max_new_tokens` and prompt length.
  - Switch to a 3B model.
  - Ensure reference model remains on CPU.

## Checkpoints
- Only LoRA adapters are saved (base model weights remain frozen).
- Reload by pointing to the saved adapter directory in a follow-up script if needed.

## Notes
- This demo is for local dev. For stability on Windows, WSL2 is recommended. Native Windows works if your CUDA + bitsandbytes install is OK.
