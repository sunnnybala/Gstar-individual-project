from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Base model (start with 3B for lower VRAM; can switch to 7B later)
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"

    # Quantization / memory
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    compute_dtype: str = "bfloat16"  # or "float16" if GPU lacks bf16

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )

    # Generation
    answer_temperature: float = 0.2
    quiz_temperature: float = 0.3
    top_p: float = 0.9
    max_new_tokens: int = 256
    max_prompt_tokens: int = 512

    # PPO
    learning_rate: float = 1e-5
    ppo_epochs: int = 1
    batch_size: int = 1
    target_kl: float = 0.1
    cliprange: float = 0.2

    # Checkpointing / logging
    save_every_n_steps: int = 20
    output_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    # Local model cache base directory (models will be stored under a subfolder per repo id)
    models_cache_dir: str = "./models_cache"

    # Experiment control
    experiment_mode: str = "style_probe"  # values: "tutor" | "style_probe"
    auto_run: bool = True  # if True, run without CLI input

    # Auto questions (used when auto_run=True or in style_probe)
    auto_questions: tuple = (
        "Explain Newton's third law simply.",
        "What is photosynthesis?", 
        "Describe the water cycle.",
        "What causes seasons on Earth?",
        "Define inertia with a short explanation.",
        "What is energy conservation?",
        "Difference between speed and velocity?",
        "Explain gravity to a child.",
        "What is an atom?",
        "How do vaccines work?",
        "What is probability in simple terms?",
        "Explain fractions with a quick note.",
        "What is a hypothesis?",
        "Why is the sky blue?",
        "What causes tides?",
    )

    # Style-probe experiment settings
    style_labels: tuple = ("story", "example", "facts")
    style_users: tuple = ("John", "Ram", "Lily")
    style_gt: dict = {"John": "story", "Ram": "example", "Lily": "facts"}
    style_eval_every_n_steps: int = 20


cfg = TrainConfig()


