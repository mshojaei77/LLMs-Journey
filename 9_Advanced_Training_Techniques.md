# Module 9: Advanced Training Techniques

### Fine-tuning & Parameter-Efficient Techniques
- **Description**: Core concepts and efficient approaches for fine-tuning language models
- **Concepts Covered**: `learning rate scheduling`, `batch size optimization`, `gradient accumulation`, `early stopping`, `validation strategies`, `model checkpointing`, `LoRA adapters`, `QLoRA`, `prefix tuning`, `prompt tuning`, `adapter tuning`, `BitFit`, `IA3`, `soft prompts`, `parameter-efficient transfer learning`
- **Learning Resources**:
  - [Fine-Tuning Transformers](https://huggingface.co/docs/transformers/training)
  - [DataCamp Fine-tuning Tutorial](https://www.datacamp.com/tutorial/fine-tuning-large-language-models)
  - [ColPali Fine-tuning Tutorial](https://github.com/merveenoyan/smol-vision/blob/main/Finetune_ColPali.ipynb) - Efficient QLoRA fine-tuning with 4-bit quantization on 32GB VRAM
  - [How to Fine-Tune LLMs in 2024 with Hugging Face](https://philschmid.de/fine-tune-llms-in-2024-with-trl) - Comprehensive guide by Philipp Schmid
  - [How to align open LLMs in 2025 with DPO & synthetic data](https://philschmid.de/rl-with-llms-in-2025-dpo) - Advanced alignment techniques
  - [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) - Detailed guide on LoRA implementation
  - [Fine-Tune Llama 3.1 Ultra-Efficiently with Unsloth](https://towardsdatascience.com/fine-tune-llama-3-1-ultra-efficiently-with-unsloth-7196c7165bab) - Resource-efficient fine-tuning approach
  - [LoRA: Low-Rank Adaptation](https://huggingface.co/blog/lora)
  - [Memory-Efficient LoRA-FA](https://arxiv.org/abs/2308.03303)
  - [Parameter Freezing Strategies](https://arxiv.org/abs/2501.07818)
  - [DeepSeek-R1 Local Fine-tuning Guide](https://x.com/_avichawla/status/1884126766132011149)
  - [Gemma PEFT Guide](https://huggingface.co/blog/gemma-peft)
  - [Fine-tuning Research Papers]:
    - [Efficient Fine-tuning Strategies](https://arxiv.org/abs/2405.00201)
    - [Advanced Fine-tuning Techniques](https://arxiv.org/abs/2410.02062)
    - [Fine-tuning Optimization](https://arxiv.org/abs/2409.00209)
  - [LoRA From Scratch Implementation](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch?view=public&section=all) - Hands-on guide to building LoRA from ground up with hyperparameter optimization scripts

- **Notebooks**:
  - [Kaggle Gemma2 9b Unsloth notebook](https://kaggle.com/code/danielhanchen/kaggle-gemma2-9b-unsloth-notebook)
  - [Quick Gemma-2B Fine-tuning Notebook](https://colab.research.google.com/drive/12OkGVWuh2lcrokExYhskSJeKrLzdmq4T?usp=sharing)
  - [Phi-4 Finetuning Tutorial on Kaggle](https://www.kaggle.com/code/unsloth/phi-4-finetuning)
  - [Fine-tuning Gemma 2 with LoRA](https://kaggle.com/code/iamleonie/fine-tuning-gemma-2-jpn-for-yomigana-with-lora) - Resource-efficient fine-tuning on T4 GPU
- **Tools**:
  - [Hugging Face PEFT](https://huggingface.co/docs/peft)
  - [UnslothAI](https://github.com/unslothai) - 60% memory reduction for efficient fine-tuning
  - [Lightning AI](https://lightning.ai/) - Generous free tier for model training
  - [io.net](https://io.net/) - Decentralized compute resources with 90% cost savings
  - [Kaggle](https://www.kaggle.com/) - Free T4 GPU access (30 hours/week)


### Advanced Fine-tuning Techniques
- **Description**: Specialized approaches for enhancing model capabilities
- **Concepts Covered**: `direct preference optimization`, `proximal policy optimization`, `constitutional AI`, `reward modeling`, `human feedback integration`, `curriculum learning`
- **Learning Resources**:
  - [How to align open LLMs in 2025 with DPO & synthetic data](https://philschmid.de/rl-with-llms-in-2025-dpo)
  - [How to Fine-Tune LLMs in 2024 with Hugging Face](https://philschmid.de/fine-tune-llms-in-2024-with-trl)
  - [Fine-tuning Research Papers]:
    - [Multi-Task Fine-tuning](https://arxiv.org/abs/2408.03094)
    - [Few-Shot Learning Approaches](https://arxiv.org/html/2408.13296v1)
    - [Instruction Tuning Optimization](https://arxiv.org/abs/2312.10793)
    - [Safety-Aligned Fine-tuning](https://arxiv.org/abs/2406.10288)
- **Notebooks**:
  -
- **Tools**:
  - [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)


### Model Merging
- **Description**: Combine multiple fine-tuned models or merge model weights to create enhanced capabilities
- **Concepts Covered**: `weight averaging`, `model fusion`, `task composition`, `knowledge distillation`, `parameter merging`, `model ensembling`
- **Learning Resources**:
  - [Merging Language Models](https://arxiv.org/abs/2401.10597)
  - [Model Merging Techniques](https://arxiv.org/abs/2306.01708)
  - [Weight Averaging Guide](https://huggingface.co/blog/merge-models)
  - [Task Arithmetic with Language Models](https://arxiv.org/abs/2212.04089)
  - [Parameter-Efficient Model Fusion](https://arxiv.org/abs/2310.13013)
- **Notebooks**:
  -
- **Tools**:
  - [mergekit](https://github.com/cg123/mergekit) - Toolkit for merging language models
  - [LM-Model-Merger](https://github.com/lm-sys/LM-Model-Merger)
  - [HuggingFace Model Merging Tools](https://huggingface.co/spaces/huggingface-projects/Model-Merger)
  - [SLERP](https://github.com/johnsmith0031/slerp_pytorch) - Spherical linear interpolation for models
  
### Fine-tuning Datasets
- **Description**: Curated datasets for instruction tuning, alignment, and specialized task adaptation of language models.
- **Concepts Covered**: `instruction tuning`, `RLHF`, `task-specific data`, `data quality`, `prompt engineering`, `human feedback`
- **Learning Resources**:
  - [Anthropic's Constitutional AI](https://www.anthropic.com/research/constitutional)
  - [Self-Instruct Paper](https://arxiv.org/abs/2212.10560)
  - [UltraFeedback Paper](https://arxiv.org/abs/2310.01377)
  - [OpenAI's InstructGPT Paper](https://arxiv.org/abs/2203.02155)
    - [DeepSeek-R1 Local Fine-tuning Guide](https://x.com/_avichawla/status/1884126766132011149) - Step-by-step guide for fine-tuning DeepSeek-R1 locally

- **Tools**:
  - Dataset Creation:
    - [Self-Instruct](https://github.com/yizhongw/self-instruct)
    - [Argilla](https://github.com/argilla-io/argilla) - Data annotation platform
    - [LIDA](https://github.com/microsoft/LIDA) - Automatic instruction dataset generation
    - [Stanford Alpaca Tools](https://github.com/tatsu-lab/stanford_alpaca)
- **Popular Datasets**:
  - Instruction Tuning:
    - [Anthropic's Constitutional AI Dataset](https://huggingface.co/datasets/anthropic/constitutional-ai)
    - [OpenAssistant Conversations](https://huggingface.co/datasets/OpenAssistant/oasst1)
    - [Alpaca Dataset](https://github.com/tatsu-lab/stanford_alpaca)
    - [Dolly Dataset](https://huggingface.co/datasets/databricks/dolly)
    - [UltraChat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat)
  - Evaluation & Feedback:
    - [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)
    - [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
    - [OpenAI WebGPT Comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons)
  - Task-Specific:
  - **Domain-Specific Datasets**:
    - Code Generation:
      - [Synthia-Coder-v1.5-I](https://huggingface.co/datasets/migtissera/Synthia-Coder-v1.5-I) - 23.5K high-quality coding samples generated with Claude Opus
    - Medical & Healthcare:
      - [Synthetic Medical Conversations (DeepSeek V3)](https://huggingface.co/datasets/OnDeviceMedNotes/synthetic-medical-conversations-deepseek-v3) - Multilingual medical conversations dataset
      - [Synthetic Medical Conversations (Chat Format)](https://huggingface.co/datasets/MaziyarPanahi/synthetic-medical-conversations-deepseek-v3-chat) - Reformatted for chat models
  - Features:
    - Multi-language support (English, Chinese, Japanese, French, German, Danish)
    - Potential expansion to 100+ languages
    - Rapid adoption by top HuggingFace leaderboard models
    - Open-source collaboration opportunities
  - Impact:
    - Enables training of specialized medical language models
    - Supports multilingual healthcare applications
    - Demonstrates rapid community adoption and iteration

### Knowledge Distillation
- **Description**: Transfer expertise from large teacher models to smaller, efficient student models.
- **Concepts Covered**: `knowledge distillation`, `teacher-student`, `model compression`
- **Learning Resources**:
  - [Knowledge Distillation Explained](https://towardsdatascience.com/knowledge-distillation-simplified-ddc070724770)
  - [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- **Tools**:
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers)
### Reasoning Models, Reinforcement Learning and Group Relative Policy Optimization (GRPO)
- **Description**: Explore models that enhance reasoning capabilities through chain-of-thought and GRPO-based training, focusing on efficient preference learning and resource-constrained environments.
- **Concepts Covered**: `chain-of-thought`, `reasoning`, `GRPO`, `preference learning`, `reward modeling`, `group-based advantage estimation`, `resource-efficient training`, `reasoning enhancement`, `reinforcement learning`, `long context scaling`
- **Learning Resources**:
  - Theory and Deep Dives:
    - [DeepSeek R1 Reasoning Primer](https://aman.ai/primers/ai/deepseek-R1/) - Detailed analysis of MoE, MLA, MTP, GRPO, and emergent reasoning behaviors
    - [DeepSeek GRPO Paper](https://arxiv.org/pdf/2402.03300) - Original paper introducing GRPO in DeepSeek Math models
    - [DeepSeek R1 Reasoning Blog](https://unsloth.ai/blog/r1-reasoning)
    - [GRPO Explained by Yannic Kilcher](https://youtube.com/watch?v=bAWV_yrqx4w) - Comprehensive explanation of PPO, REINFORCE, KL divergence, advantages & more
    - [DeepSeek R1 Theory Overview](https://www.youtube.com/watch?v=QdEuh2UVbu0)
    - [How R1 and GRPO Work - Technical Deep Dive](https://www.youtube.com/watch?v=-7Y4s7ItQQ4)
    - [The Batch: RL in Reasoning Models](https://hubs.la/Q0351_T10)
    - [Open-R1](https://huggingface.co/blog/open-r1/update-1)
    - [Kimi k1.5 Paper](https://arxiv.org/abs/2401.12863) - Scaling Reinforcement Learning with LLMs
    - [AGIEntry Kimi Overview](https://agientry.com) - Analysis of Kimi k1.5's reasoning capabilities

  - Implementation Guides:
    - [TinyZero](https://github.com/Jiayi-Pan/TinyZero) - Berkeley researchers' $30 reproduction of DeepSeek R1's core technology
    - [GRPO Poetry Generation Notebook](https://colab.research.google.com/drive/1Ty0ovsrpw8i-zJvDhlSAtBIVw3EZfHK5?usp=sharing)
    - [Qwen 0.5B GRPO Notebook](https://colab.research.google.com/drive/1bfhs1FMLW3FGa8ydvkOZyBNxLYOu0Hev?usp=sharing)
    - [Phi-4 14B GRPO Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4_(14B)-GRPO.ipynb)
    - [Llama 3.1 8B GRPO Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
    - [GRPO Implementation for Qwen-0.5B](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) - Code implementation showing +10% accuracy gain on GSM8K

- **Key Features**:
  - Enhanced reasoning capabilities for complex domains (math, coding, science)
  - Self-improvement through reward-based learning
  - Real-time logic refinement beyond static training data
  - Resource-efficient implementation for practical deployment
  - Long context scaling for improved performance
  - State-of-the-art reasoning benchmarks (e.g., 77.5 on AIME, 96.2 on MATH 500)
  - Multi-modal capabilities with text, images, and file analysis
  - Web search integration across 100+ sites

- **Tools**:
  - [Unsloth](https://github.com/unslothai/unsloth) - Optimized GRPO implementation
  - [DeepSeek-R1 Training Framework](https://github.com/deepseek-ai/DeepSeek-R1) - Reference implementation
  - [Kimi.ai](https://kimi.ai) - Free web-based reasoning model with unlimited usage

- **Security Considerations**:
  - Implement robust safety measures to prevent reasoning step exploitation
  - Monitor and validate reasoning chains for potential vulnerabilities
  - Regular security audits of model outputs and reasoning patterns

### GRPO Datasets
- **Description**: Curated datasets for training and evaluating GRPO-based models, with focus on reasoning, poetry, and domain-specific tasks.
- **Concepts Covered**: `dataset curation`, `chain-of-thought patterns`, `reasoning verification`, `poetry generation`, `scientific problem-solving`, `data preprocessing`, `quality filtering`
- **Learning Resources**:
  - [Guide to Creating CoT Datasets](https://huggingface.co/blog/creating-chain-of-thought-datasets)
  - [Data Generation with R1 Models](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/data_generation.md)
  - [Verse Dataset Creation Tutorial](https://github.com/PleIAs/verse-wikisource/blob/main/TUTORIAL.md)
  - [Scientific Dataset Curation Guide](https://github.com/EricLu1/SCP-Guide)
- **Datasets**:
  - Poetry & Creative:
    - [PleIAs Verse Wikisource](https://huggingface.co/datasets/PleIAs/verse-wikisource) - 200,000 verses for poetry training
  
  - Reasoning & Chain-of-Thought:
    - [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) - High-quality CoT with generation code
    - [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) - Comprehensive reasoning patterns distilled from R1
    - [Evalchemy Dataset](https://huggingface.co/datasets/evalchemy) - Complementary reasoning dataset
    - [R1-Distill-SFT](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT) - 1.8M samples from DeepSeek-R1-32b
  
  - Domain-Specific:
    - [Sky-T1_data_17k](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k) - 17k verified samples for coding, math, science
    - [SCP-116K](https://huggingface.co/datasets/EricLu/SCP-116K) - Scientific problem-solving (Physics, Chemistry, Biology)
    - [FineQwQ-142k](https://huggingface.co/datasets/qingy2024/FineQwQ-142k) - Math, Coding, General reasoning
  
  - Combined & Reformatted:
    - [Dolphin-R1](https://huggingface.co/datasets/cognitivecomputations/dolphin-r1) - Combined R1 and Gemini 2 reasoning
    - [Dolphin-R1-DeepSeek](https://huggingface.co/datasets/mlabonne/dolphin-r1-deepseek) - DeepSeek-compatible format
    - [Dolphin-R1-Flash](https://huggingface.co/datasets/mlabonne/dolphin-r1-flash) - Flash Thinking format