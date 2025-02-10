# Module 9: Advanced Training Techniques

### Fine-tuning & Parameter-Efficient Techniques
- **Description**: Core concepts and efficient approaches for fine-tuning language models
- **Concepts Covered**: `learning rate scheduling`, `batch size optimization`, `gradient accumulation`, `early stopping`, `validation strategies`, `model checkpointing`, `LoRA adapters`, `QLoRA`, `prefix tuning`, `prompt tuning`, `adapter tuning`, `BitFit`, `IA3`, `soft prompts`, `parameter-efficient transfer learning`
- **Learning Resources**:
  - [![Fine-Tuning Transformers](https://badgen.net/badge/Docs/Fine-Tuning%20Transformers/green)](https://huggingface.co/docs/transformers/training)
  - [![DataCamp Fine-tuning Tutorial](https://badgen.net/badge/Tutorial/DataCamp%20Fine-tuning%20Tutorial/blue)](https://www.datacamp.com/tutorial/fine-tuning-large-language-models)
  - [![ColPali Fine-tuning Tutorial](https://badgen.net/badge/Tutorial/ColPali%20Fine-tuning%20Tutorial/blue)](https://github.com/merveenoyan/smol-vision/blob/main/Finetune_ColPali.ipynb) - Efficient QLoRA fine-tuning with 4-bit quantization on 32GB VRAM
  - [![How to Fine-Tune LLMs in 2024 with Hugging Face](https://badgen.net/badge/Blog/How%20to%20Fine-Tune%20LLMs%20in%202024%20with%20Hugging%20Face/cyan)](https://philschmid.de/fine-tune-llms-in-2024-with-trl) - Comprehensive guide by Philipp Schmid
  - [![How to align open LLMs in 2025 with DPO & synthetic data](https://badgen.net/badge/Blog/How%20to%20align%20open%20LLMs%20in%202025%20with%20DPO%20%26%20synthetic%20data/cyan)](https://philschmid.de/rl-with-llms-in-2025-dpo) - Advanced alignment techniques
  - [![Practical Tips for Finetuning LLMs Using LoRA](https://badgen.net/badge/Blog/Practical%20Tips%20for%20Finetuning%20LLMs%20Using%20LoRA/cyan)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) - Detailed guide on LoRA implementation
  - [![Fine-Tune Llama 3.1 Ultra-Efficiently with Unsloth](https://badgen.net/badge/Blog/Fine-Tune%20Llama%203.1%20Ultra-Efficiently%20with%20Unsloth/cyan)](https://towardsdatascience.com/fine-tune-llama-3-1-ultra-efficiently-with-unsloth-7196c7165bab) - Resource-efficient fine-tuning approach
  - [![LoRA: Low-Rank Adaptation](https://badgen.net/badge/Blog/LoRA:%20Low-Rank%20Adaptation/cyan)](https://huggingface.co/blog/lora)
  - [![Memory-Efficient LoRA-FA](https://badgen.net/badge/Paper/Memory-Efficient%20LoRA-FA/purple)](https://arxiv.org/abs/2308.03303)
  - [![Parameter Freezing Strategies](https://badgen.net/badge/Paper/Parameter%20Freezing%20Strategies/purple)](https://arxiv.org/abs/2501.07818)
  - [![DeepSeek-R1 Local Fine-tuning Guide](https://badgen.net/badge/Website/DeepSeek-R1%20Local%20Fine-tuning%20Guide/blue)](https://x.com/_avichawla/status/1884126766132011149)
  - [![Gemma PEFT Guide](https://badgen.net/badge/Blog/Gemma%20PEFT%20Guide/cyan)](https://huggingface.co/blog/gemma-peft)
  - [Fine-tuning Research Papers]:
    - [![Efficient Fine-tuning Strategies](https://badgen.net/badge/Paper/Efficient%20Fine-tuning%20Strategies/purple)](https://arxiv.org/abs/2405.00201)
    - [![Advanced Fine-tuning Techniques](https://badgen.net/badge/Paper/Advanced%20Fine-tuning%20Techniques/purple)](https://arxiv.org/abs/2410.02062)
    - [![Fine-tuning Optimization](https://badgen.net/badge/Paper/Fine-tuning%20Optimization/purple)](https://arxiv.org/abs/2409.00209)
  - [![LoRA From Scratch Implementation](https://badgen.net/badge/Tutorial/LoRA%20From%20Scratch%20Implementation/blue)](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch?view=public&section=all) - Hands-on guide to building LoRA from ground up with hyperparameter optimization scripts

- **Notebooks**:
  - [![Kaggle Gemma2 9b Unsloth notebook](https://badgen.net/badge/Colab%20Notebook/Kaggle%20Gemma2%209b%20Unsloth%20notebook/orange)](https://kaggle.com/code/danielhanchen/kaggle-gemma2-9b-unsloth-notebook)
  - [![Quick Gemma-2B Fine-tuning Notebook](https://badgen.net/badge/Colab%20Notebook/Quick%20Gemma-2B%20Fine-tuning%20Notebook/orange)](https://colab.research.google.com/drive/12OkGVWuh2lcrokExYhskSJeKrLzdmq4T?usp=sharing)
  - [![Phi-4 Finetuning Tutorial on Kaggle](https://badgen.net/badge/Colab%20Notebook/Phi-4%20Finetuning%20Tutorial%20on%20Kaggle/orange)](https://www.kaggle.com/code/unsloth/phi-4-finetuning)
  - [![Fine-tuning Gemma 2 with LoRA](https://badgen.net/badge/Colab%20Notebook/Fine-tuning%20Gemma%202%20with%20LoRA/orange)](https://kaggle.com/code/iamleonie/fine-tuning-gemma-2-jpn-for-yomigana-with-lora) - Resource-efficient fine-tuning on T4 GPU
- **Tools**:
  - [![Hugging Face PEFT](https://badgen.net/badge/Framework/Hugging%20Face%20PEFT/green)](https://huggingface.co/docs/peft)
  - [![UnslothAI](https://badgen.net/badge/Github%20Repository/UnslothAI/gray)](https://github.com/unslothai) - 60% memory reduction for efficient fine-tuning
  - [![Lightning AI](https://badgen.net/badge/Framework/Lightning%20AI/green)](https://lightning.ai/) - Generous free tier for model training
  - [![io.net](https://badgen.net/badge/API%20Provider/io.net/blue)](https://io.net/) - Decentralized compute resources with 90% cost savings
  - [![Kaggle](https://badgen.net/badge/Website/Kaggle/blue)](https://www.kaggle.com/) - Free T4 GPU access (30 hours/week)


### Advanced Fine-tuning Techniques
- **Description**: Specialized approaches for enhancing model capabilities
- **Concepts Covered**: `direct preference optimization`, `proximal policy optimization`, `constitutional AI`, `reward modeling`, `human feedback integration`, `curriculum learning`
- **Learning Resources**:
  - [![How to align open LLMs in 2025 with DPO & synthetic data](https://badgen.net/badge/Blog/How%20to%20align%20open%20LLMs%20in%202025%20with%20DPO%20%26%20synthetic%20data/cyan)](https://philschmid.de/rl-with-llms-in-2025-dpo)
  - [![How to Fine-Tune LLMs in 2024 with Hugging Face](https://badgen.net/badge/Blog/How%20to%20Fine-Tune%20LLMs%20in%202024%20with%20Hugging%20Face/cyan)](https://philschmid.de/fine-tune-llms-in-2024-with-trl)
  - [Fine-tuning Research Papers]:
    - [![Multi-Task Fine-tuning](https://badgen.net/badge/Paper/Multi-Task%20Fine-tuning/purple)](https://arxiv.org/abs/2408.03094)
    - [![Few-Shot Learning Approaches](https://badgen.net/badge/Paper/Few-Shot%20Learning%20Approaches/purple)](https://arxiv.org/html/2408.13296v1)
    - [![Instruction Tuning Optimization](https://badgen.net/badge/Paper/Instruction%20Tuning%20Optimization/purple)](https://arxiv.org/abs/2312.10793)
    - [![Safety-Aligned Fine-tuning](https://badgen.net/badge/Paper/Safety-Aligned%20Fine-tuning/purple)](https://arxiv.org/abs/2406.10288)
- **Notebooks**:
  -
- **Tools**:
  - [![TRL (Transformer Reinforcement Learning)](https://badgen.net/badge/Github%20Repository/TRL%20(Transformer%20Reinforcement%20Learning)/gray)](https://github.com/huggingface/trl)


### Model Merging
- **Description**: Combine multiple fine-tuned models or merge model weights to create enhanced capabilities
- **Concepts Covered**: `weight averaging`, `model fusion`, `task composition`, `knowledge distillation`, `parameter merging`, `model ensembling`
- **Learning Resources**:
  - [![Merging Language Models](https://badgen.net/badge/Paper/Merging%20Language%20Models/purple)](https://arxiv.org/abs/2401.10597)
  - [![Model Merging Techniques](https://badgen.net/badge/Paper/Model%20Merging%20Techniques/purple)](https://arxiv.org/abs/2306.01708)
  - [![Weight Averaging Guide](https://badgen.net/badge/Blog/Weight%20Averaging%20Guide/cyan)](https://huggingface.co/blog/merge-models)
  - [![Task Arithmetic with Language Models](https://badgen.net/badge/Paper/Task%20Arithmetic%20with%20Language%20Models/purple)](https://arxiv.org/abs/2212.04089)
  - [![Parameter-Efficient Model Fusion](https://badgen.net/badge/Paper/Parameter-Efficient%20Model%20Fusion/purple)](https://arxiv.org/abs/2310.13013)
- **Notebooks**:
  -
- **Tools**:
  - [![mergekit](https://badgen.net/badge/Github%20Repository/mergekit/gray)](https://github.com/cg123/mergekit) - Toolkit for merging language models
  - [![LM-Model-Merger](https://badgen.net/badge/Github%20Repository/LM-Model-Merger/gray)](https://github.com/lm-sys/LM-Model-Merger)
  - [![HuggingFace Model Merging Tools](https://badgen.net/badge/Hugging%20Face%20Space/HuggingFace%20Model%20Merging%20Tools/yellow)](https://huggingface.co/spaces/huggingface-projects/Model-Merger)
  - [![SLERP](https://badgen.net/badge/Github%20Repository/SLERP/gray)](https://github.com/johnsmith0031/slerp_pytorch) - Spherical linear interpolation for models
  
### Fine-tuning Datasets
- **Description**: Curated datasets for instruction tuning, alignment, and specialized task adaptation of language models.
- **Concepts Covered**: `instruction tuning`, `RLHF`, `task-specific data`, `data quality`, `prompt engineering`, `human feedback`
- **Learning Resources**:
  - [![Anthropic's Constitutional AI](https://badgen.net/badge/Website/Anthropic's%20Constitutional%20AI/blue)](https://www.anthropic.com/research/constitutional)
  - [![Self-Instruct Paper](https://badgen.net/badge/Paper/Self-Instruct%20Paper/purple)](https://arxiv.org/abs/2212.10560)
  - [![UltraFeedback Paper](https://badgen.net/badge/Paper/UltraFeedback%20Paper/purple)](https://arxiv.org/abs/2310.01377)
  - [![OpenAI's InstructGPT Paper](https://badgen.net/badge/Paper/OpenAI's%20InstructGPT%20Paper/purple)](https://arxiv.org/abs/2203.02155)
    - [![DeepSeek-R1 Local Fine-tuning Guide](https://badgen.net/badge/Website/DeepSeek-R1%20Local%20Fine-tuning%20Guide/blue)](https://x.com/_avichawla/status/1884126766132011149) - Step-by-step guide for fine-tuning DeepSeek-R1 locally

- **Tools**:
  - Dataset Creation:
    - [![Self-Instruct](https://badgen.net/badge/Github%20Repository/Self-Instruct/gray)](https://github.com/yizhongw/self-instruct)
    - [![Argilla](https://badgen.net/badge/Github%20Repository/Argilla/gray)](https://github.com/argilla-io/argilla) - Data annotation platform
    - [![LIDA](https://badgen.net/badge/Github%20Repository/LIDA/gray)](https://github.com/microsoft/LIDA) - Automatic instruction dataset generation
    - [![Stanford Alpaca Tools](https://badgen.net/badge/Github%20Repository/Stanford%20Alpaca%20Tools/gray)](https://github.com/tatsu-lab/stanford_alpaca)
- **Popular Datasets**:
  - Instruction Tuning:
    - [![Anthropic's Constitutional AI Dataset](https://badgen.net/badge/Hugging%20Face%20Dataset/Anthropic's%20Constitutional%20AI%20Dataset/yellow)](https://huggingface.co/datasets/anthropic/constitutional-ai)
    - [![OpenAssistant Conversations](https://badgen.net/badge/Hugging%20Face%20Dataset/OpenAssistant%20Conversations/yellow)](https://huggingface.co/datasets/OpenAssistant/oasst1)
    - [![Alpaca Dataset](https://badgen.net/badge/Github%20Repository/Alpaca%20Dataset/gray)](https://github.com/tatsu-lab/stanford_alpaca)
    - [![Dolly Dataset](https://badgen.net/badge/Hugging%20Face%20Dataset/Dolly%20Dataset/yellow)](https://huggingface.co/datasets/databricks/dolly)
    - [![UltraChat](https://badgen.net/badge/Hugging%20Face%20Dataset/UltraChat/yellow)](https://huggingface.co/datasets/HuggingFaceH4/ultrachat)
  - Evaluation & Feedback:
    - [![UltraFeedback](https://badgen.net/badge/Hugging%20Face%20Dataset/UltraFeedback/yellow)](https://huggingface.co/datasets/openbmb/UltraFeedback)
    - [![Anthropic HH-RLHF](https://badgen.net/badge/Hugging%20Face%20Dataset/Anthropic%20HH-RLHF/yellow)](https://huggingface.co/datasets/Anthropic/hh-rlhf)
    - [![OpenAI WebGPT Comparisons](https://badgen.net/badge/Hugging%20Face%20Dataset/OpenAI%20WebGPT%20Comparisons/yellow)](https://huggingface.co/datasets/openai/webgpt_comparisons)
  - Task-Specific:
  - **Domain-Specific Datasets**:
    - Code Generation:
      - [![Synthia-Coder-v1.5-I](https://badgen.net/badge/Hugging%20Face%20Dataset/Synthia-Coder-v1.5-I/yellow)](https://huggingface.co/datasets/migtissera/Synthia-Coder-v1.5-I) - 23.5K high-quality coding samples generated with Claude Opus
    - Medical & Healthcare:
      - [![Synthetic Medical Conversations (DeepSeek V3)](https://badgen.net/badge/Hugging%20Face%20Dataset/Synthetic%20Medical%20Conversations%20(DeepSeek%20V3)/yellow)](https://huggingface.co/datasets/OnDeviceMedNotes/synthetic-medical-conversations-deepseek-v3) - Multilingual medical conversations dataset
      - [![Synthetic Medical Conversations (Chat Format)](https://badgen.net/badge/Hugging%20Face%20Dataset/Synthetic%20Medical%20Conversations%20(Chat%20Format)/yellow)](https://huggingface.co/datasets/MaziyarPanahi/synthetic-medical-conversations-deepseek-v3-chat) - Reformatted for chat models
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
  - [![Knowledge Distillation Explained](https://badgen.net/badge/Tutorial/Knowledge%20Distillation%20Explained/blue)](https://towardsdatascience.com/knowledge-distillation-simplified-ddc070724770)
  - [![DistilBERT Paper](https://badgen.net/badge/Paper/DistilBERT%20Paper/purple)](https://arxiv.org/abs/1910.01108)
- **Tools**:
  - [![Hugging Face Transformers](https://badgen.net/badge/Framework/Hugging%20Face%20Transformers/green)](https://huggingface.co/docs/transformers)
### Reasoning Models, Reinforcement Learning and Group Relative Policy Optimization (GRPO)
- **Description**: Explore models that enhance reasoning capabilities through chain-of-thought and GRPO-based training, focusing on efficient preference learning and resource-constrained environments.
- **Concepts Covered**: `chain-of-thought`, `reasoning`, `GRPO`, `preference learning`, `reward modeling`, `group-based advantage estimation`, `resource-efficient training`, `reasoning enhancement`, `reinforcement learning`, `long context scaling`
- **Learning Resources**:
  - Theory and Deep Dives:
    - [![DeepSeek R1 Reasoning Primer](https://badgen.net/badge/Blog/DeepSeek%20R1%20Reasoning%20Primer/cyan)](https://aman.ai/primers/ai/deepseek-R1/) - Detailed analysis of MoE, MLA, MTP, GRPO, and emergent reasoning behaviors
    - [![DeepSeek GRPO Paper](https://badgen.net/badge/Paper/DeepSeek%20GRPO%20Paper/purple)](https://arxiv.org/pdf/2402.03300) - Original paper introducing GRPO in DeepSeek Math models
    - [![DeepSeek R1 Reasoning Blog](https://badgen.net/badge/Blog/DeepSeek%20R1%20Reasoning%20Blog/cyan)](https://unsloth.ai/blog/r1-reasoning)
    - [![GRPO Explained by Yannic Kilcher](https://badgen.net/badge/Video/GRPO%20Explained%20by%20Yannic%20Kilcher/red)](https://youtube.com/watch?v=bAWV_yrqx4w) - Comprehensive explanation of PPO, REINFORCE, KL divergence, advantages & more
    - [![DeepSeek R1 Theory Overview](https://badgen.net/badge/Video/DeepSeek%20R1%20Theory%20Overview/red)](https://www.youtube.com/watch?v=QdEuh2UVbu0)
    - [![How R1 and GRPO Work - Technical Deep Dive](https://badgen.net/badge/Video/How%20R1%20and%20GRPO%20Work%20-%20Technical%20Deep%20Dive/red)](https://www.youtube.com/watch?v=-7Y4s7ItQQ4)
    - [![The Batch: RL in Reasoning Models](https://badgen.net/badge/Website/The%20Batch:%20RL%20in%20Reasoning%20Models/blue)](https://hubs.la/Q0351_T10)
    - [![Open-R1](https://badgen.net/badge/Blog/Open-R1/cyan)](https://huggingface.co/blog/open-r1/update-1)
    - [![Kimi k1.5 Paper](https://badgen.net/badge/Paper/Kimi%20k1.5%20Paper/purple)](https://arxiv.org/abs/2401.12863) - Scaling Reinforcement Learning with LLMs
    - [![AGIEntry Kimi Overview](https://badgen.net/badge/Website/AGIEntry%20Kimi%20Overview/blue)](https://agientry.com) - Analysis of Kimi k1.5's reasoning capabilities

  - Implementation Guides:
    - [![TinyZero](https://badgen.net/badge/Github%20Repository/TinyZero/gray)](https://github.com/Jiayi-Pan/TinyZero) - Berkeley researchers' $30 reproduction of DeepSeek R1's core technology
    - [![GRPO Poetry Generation Notebook](https://badgen.net/badge/Colab%20Notebook/GRPO%20Poetry%20Generation%20Notebook/orange)](https://colab.research.google.com/drive/1Ty0ovsrpw8i-zJvDhlSAtBIVw3EZfHK5?usp=sharing)
    - [![Qwen 0.5B GRPO Notebook](https://badgen.net/badge/Colab%20Notebook/Qwen%200.5B%20GRPO%20Notebook/orange)](https://colab.research.google.com/drive/1bfhs1FMLW3FGa8ydvkOZyBNxLYOu0Hev?usp=sharing)
    - [![Phi-4 14B GRPO Notebook](https://badgen.net/badge/Colab%20Notebook/Phi-4%2014B%20GRPO%20Notebook/orange)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4_(14B)-GRPO.ipynb)
    - [![Llama 3.1 8B GRPO Notebook](https://badgen.net/badge/Colab%20Notebook/Llama%203.1%208B%20GRPO%20Notebook/orange)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
    - [![GRPO Implementation for Qwen-0.5B](https://badgen.net/badge/Website/GRPO%20Implementation%20for%20Qwen-0.5B/blue)](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) - Code implementation showing +10% accuracy gain on GSM8K

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
  - [![Unsloth](https://badgen.net/badge/Github%20Repository/Unsloth/gray)](https://github.com/unslothai/unsloth) - Optimized GRPO implementation
  - [![DeepSeek-R1 Training Framework](https://badgen.net/badge/Github%20Repository/DeepSeek-R1%20Training%20Framework/gray)](https://github.com/deepseek-ai/DeepSeek-R1) - Reference implementation
  - [![Kimi.ai](https://badgen.net/badge/Website/Kimi.ai/blue)](https://kimi.ai) - Free web-based reasoning model with unlimited usage

- **Security Considerations**:
  - Implement robust safety measures to prevent reasoning step exploitation
  - Monitor and validate reasoning chains for potential vulnerabilities
  - Regular security audits of model outputs and reasoning patterns

### GRPO Datasets
- **Description**: Curated datasets for training and evaluating GRPO-based models, with focus on reasoning, poetry, and domain-specific tasks.
- **Concepts Covered**: `dataset curation`, `chain-of-thought patterns`, `reasoning verification`, `poetry generation`, `scientific problem-solving`, `data preprocessing`, `quality filtering`
- **Learning Resources**:
  - [![Guide to Creating CoT Datasets](https://badgen.net/badge/Blog/Guide%20to%20Creating%20CoT%20Datasets/cyan)](https://huggingface.co/blog/creating-chain-of-thought-datasets)
  - [![Data Generation with R1 Models](https://badgen.net/badge/Github%20Repository/Data%20Generation%20with%20R1%20Models/gray)](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/data_generation.md)
  - [![Verse Dataset Creation Tutorial](https://badgen.net/badge/Tutorial/Verse%20Dataset%20Creation%20Tutorial/blue)](https://github.com/PleIAs/verse-wikisource/blob/main/TUTORIAL.md)
  - [![Scientific Dataset Curation Guide](https://badgen.net/badge/Github%20Repository/Scientific%20Dataset%20Curation%20Guide/gray)](https://github.com/EricLu1/SCP-Guide)
- **Datasets**:
  - Poetry & Creative:
    - [![PleIAs Verse Wikisource](https://badgen.net/badge/Hugging%20Face%20Dataset/PleIAs%20Verse%20Wikisource/yellow)](https://huggingface.co/datasets/PleIAs/verse-wikisource) - 200,000 verses for poetry training
  
  - Reasoning & Chain-of-Thought:
    - [![Bespoke-Stratos-17k](https://badgen.net/badge/Hugging%20Face%20Dataset/Bespoke-Stratos-17k/yellow)](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) - High-quality CoT with generation code
    - [![OpenThoughts-114k](https://badgen.net/badge/Hugging%20Face%20Dataset/OpenThoughts-114k/yellow)](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) - Comprehensive reasoning patterns distilled from R1
    - [![Evalchemy Dataset](https://badgen.net/badge/Hugging%20Face%20Dataset/Evalchemy%20Dataset/yellow)](https://huggingface.co/datasets/evalchemy) - Complementary reasoning dataset
    - [![R1-Distill-SFT](https://badgen.net/badge/Hugging%20Face%20Dataset/R1-Distill-SFT/yellow)](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT) - 1.8M samples from DeepSeek-R1-32b
  
  - Domain-Specific:
    - [![Sky-T1_data_17k](https://badgen.net/badge/Hugging%20Face%20Dataset/Sky-T1_data_17k/yellow)](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k) - 17k verified samples for coding, math, science
    - [![SCP-116K](https://badgen.net/badge/Hugging%20Face%20Dataset/SCP-116K/yellow)](https://huggingface.co/datasets/EricLu/SCP-116K) - Scientific problem-solving (Physics, Chemistry, Biology)
    - [![FineQwQ-142k](https://badgen.net/badge/Hugging%20Face%20Dataset/FineQwQ-142k/yellow)](https://huggingface.co/datasets/qingy2024/FineQwQ-142k) - Math, Coding, General reasoning
  
  - Combined & Reformatted:
    - [![Dolphin-R1](https://badgen.net/badge/Hugging%20Face%20Dataset/Dolphin-R1/yellow)](https://huggingface.co/datasets/cognitivecomputations/dolphin-r1) - Combined R1 and Gemini 2 reasoning
    - [![Dolphin-R1-DeepSeek](https://badgen.net/badge/Hugging%20Face%20Dataset/Dolphin-R1-DeepSeek/yellow)](https://huggingface.co/datasets/mlabonne/dolphin-r1-deepseek) - DeepSeek-compatible format
    - [![Dolphin-R1-Flash](https://badgen.net/badge/Hugging%20Face%20Dataset/Dolphin-R1-Flash/yellow)](https://huggingface.co/datasets/mlabonne/dolphin-r1-flash) - Flash Thinking format