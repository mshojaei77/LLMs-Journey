# Module 9: Post Training Techniques

### Fine-tuning & Parameter-Efficient Techniques
- **Description**: Core concepts and efficient approaches for fine-tuning language models
- **Concepts Covered**: `learning rate scheduling`, `batch size optimization`, `gradient accumulation`, `early stopping`, `validation strategies`, `model checkpointing`, `LoRA adapters`, `QLoRA`, `prefix tuning`, `prompt tuning`, `adapter tuning`, `BitFit`, `IA3`, `soft prompts`, `parameter-efficient transfer learning`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Fine-Tuning Transformers](https://badgen.net/badge/Docs/Fine-Tuning%20Transformers/green)](https://huggingface.co/docs/transformers/training) | [![DataCamp Fine-tuning Tutorial](https://badgen.net/badge/Tutorial/DataCamp%20Fine-tuning%20Tutorial/blue)](https://www.datacamp.com/tutorial/fine-tuning-large-language-models) |
| [![How to Fine-Tune LLMs in 2024 with Hugging Face](https://badgen.net/badge/Blog/How%20to%20Fine-Tune%20LLMs%20in%202024%20with%20Hugging%20Face/pink)](https://philschmid.de/fine-tune-llms-in-2024-with-trl) | [![ColPali Fine-tuning Tutorial](https://badgen.net/badge/Tutorial/ColPali%20Fine-tuning%20Tutorial/blue)](https://github.com/merveenoyan/smol-vision/blob/main/Finetune_ColPali.ipynb) |
| [![LoRA: Low-Rank Adaptation](https://badgen.net/badge/Blog/LoRA:%20Low-Rank%20Adaptation/pink)](https://huggingface.co/blog/lora) | [![Parameter Freezing Strategies](https://badgen.net/badge/Paper/Parameter%20Freezing%20Strategies/purple)](https://arxiv.org/abs/2501.07818) |
| [![Practical Tips for Finetuning LLMs Using LoRA](https://badgen.net/badge/Blog/Practical%20Tips%20for%20Finetuning%20LLMs%20Using%20LoRA/pink)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) | [![Memory-Efficient LoRA-FA](https://badgen.net/badge/Paper/Memory-Efficient%20LoRA-FA/purple)](https://arxiv.org/abs/2308.03303) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Hugging Face PEFT](https://badgen.net/badge/Framework/Hugging%20Face%20PEFT/green)](https://huggingface.co/docs/peft) | [![UnslothAI](https://badgen.net/badge/Github%20Repository/UnslothAI/cyan)](https://github.com/unslothai) |
| [![Lightning AI](https://badgen.net/badge/Framework/Lightning%20AI/green)](https://lightning.ai/) | [![io.net](https://badgen.net/badge/API%20Provider/io.net/blue)](https://io.net/) |
| [![Kaggle](https://badgen.net/badge/Website/Kaggle/blue)](https://www.kaggle.com/) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Kaggle Gemma2 9b Unsloth notebook](https://badgen.net/badge/Colab%20Notebook/Kaggle%20Gemma2%209b%20Unsloth%20notebook/orange)](https://kaggle.com/code/danielhanchen/kaggle-gemma2-9b-unsloth-notebook) | Fine-tune Gemma 2 9B with Unsloth optimization |
| [![Quick Gemma-2B Fine-tuning Notebook](https://badgen.net/badge/Colab%20Notebook/Quick%20Gemma-2B%20Fine-tuning%20Notebook/orange)](https://colab.research.google.com/drive/12OkGVWuh2lcrokExYhskSJeKrLzdmq4T?usp=sharing) | Basic Gemma 2B fine-tuning example |
| [![Phi-4 Finetuning Tutorial](https://badgen.net/badge/Colab%20Notebook/Phi-4%20Finetuning%20Tutorial%20on%20Kaggle/orange)](https://www.kaggle.com/code/unsloth/phi-4-finetuning) | Fine-tune Phi-4 model efficiently |
| [![Fine-tuning Gemma 2 with LoRA](https://badgen.net/badge/Colab%20Notebook/Fine-tuning%20Gemma%202%20with%20LoRA/orange)](https://kaggle.com/code/iamleonie/fine-tuning-gemma-2-jpn-for-yomigana-with-lora) | LoRA fine-tuning on T4 GPU |

### Advanced Fine-tuning Techniques
- **Description**: Specialized approaches for enhancing model capabilities
- **Concepts Covered**: `direct preference optimization`, `proximal policy optimization`, `constitutional AI`, `reward modeling`, `human feedback integration`, `curriculum learning`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![How to align open LLMs in 2025 with DPO & synthetic data](https://badgen.net/badge/Blog/How%20to%20align%20open%20LLMs%20in%202025%20with%20DPO%20%26%20synthetic%20data/pink)](https://philschmid.de/rl-with-llms-in-2025-dpo) | [![Multi-Task Fine-tuning](https://badgen.net/badge/Paper/Multi-Task%20Fine-tuning/purple)](https://arxiv.org/abs/2408.03094) |
| [![How to Fine-Tune LLMs in 2024 with Hugging Face](https://badgen.net/badge/Blog/How%20to%20Fine-Tune%20LLMs%20in%202024%20with%20Hugging%20Face/pink)](https://philschmid.de/fine-tune-llms-in-2024-with-trl) | [![Few-Shot Learning Approaches](https://badgen.net/badge/Paper/Few-Shot%20Learning%20Approaches/purple)](https://arxiv.org/html/2408.13296v1) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![TRL (Transformer Reinforcement Learning)](https://badgen.net/badge/Github%20Repository/TRL%20(Transformer%20Reinforcement%20Learning)/cyan)](https://github.com/huggingface/trl) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Knowledge Distillation Basics](https://badgen.net/badge/Notebook/Knowledge%20Distillation%20Basics/orange)](notebooks/knowledge_distillation_basics.ipynb) | Implement basic knowledge distillation |
| [![Student Model Training](https://badgen.net/badge/Notebook/Student%20Model%20Training/orange)](notebooks/student_model_training.ipynb) | Train efficient student models |

### Model Merging
- **Description**: Combine multiple fine-tuned models or merge model weights to create enhanced capabilities
- **Concepts Covered**: `weight averaging`, `model fusion`, `task composition`, `knowledge distillation`, `parameter merging`, `model ensembling`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Merging Language Models](https://badgen.net/badge/Paper/Merging%20Language%20Models/purple)](https://arxiv.org/abs/2401.10597) | [![Task Arithmetic with Language Models](https://badgen.net/badge/Paper/Task%20Arithmetic%20with%20Language%20Models/purple)](https://arxiv.org/abs/2212.04089) |
| [![Weight Averaging Guide](https://badgen.net/badge/Blog/Weight%20Averaging%20Guide/pink)](https://huggingface.co/blog/merge-models) | [![Parameter-Efficient Model Fusion](https://badgen.net/badge/Paper/Parameter-Efficient%20Model%20Fusion/purple)](https://arxiv.org/abs/2310.13013) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![mergekit](https://badgen.net/badge/Github%20Repository/mergekit/cyan)](https://github.com/cg123/mergekit) | [![SLERP](https://badgen.net/badge/Github%20Repository/SLERP/cyan)](https://github.com/johnsmith0031/slerp_pytorch) |
| [![LM-Model-Merger](https://badgen.net/badge/Github%20Repository/LM-Model-Merger/cyan)](https://github.com/lm-sys/LM-Model-Merger) | [![HuggingFace Model Merging Tools](https://badgen.net/badge/Hugging%20Face%20Space/HuggingFace%20Model%20Merging%20Tools/yellow)](https://huggingface.co/spaces/huggingface-projects/Model-Merger) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Model Merging Basics](https://badgen.net/badge/Notebook/Model%20Merging%20Basics/orange)](notebooks/model_merging_basics.ipynb) | Basic model merging techniques |
| [![Weight Averaging](https://badgen.net/badge/Notebook/Weight%20Averaging/orange)](notebooks/weight_averaging.ipynb) | Implement weight averaging |
| [![Model Fusion](https://badgen.net/badge/Notebook/Model%20Fusion/orange)](notebooks/model_fusion.ipynb) | Advanced model fusion techniques |

### Fine-tuning Datasets
- **Description**: Curated datasets for instruction tuning, alignment, and specialized task adaptation of language models.
- **Concepts Covered**: `instruction tuning`, `RLHF`, `task-specific data`, `data quality`, `prompt engineering`, `human feedback`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Anthropic's Constitutional AI](https://badgen.net/badge/Website/Anthropic's%20Constitutional%20AI/blue)](https://www.anthropic.com/research/constitutional) - Core principles of aligned fine-tuning | [![Self-Instruct Paper](https://badgen.net/badge/Paper/Self-Instruct%20Paper/purple)](https://arxiv.org/abs/2212.10560) - Automated instruction generation |
| [![OpenAI's InstructGPT Paper](https://badgen.net/badge/Paper/OpenAI's%20InstructGPT%20Paper/purple)](https://arxiv.org/abs/2203.02155) - Foundational RLHF approach | [![UltraFeedback Paper](https://badgen.net/badge/Paper/UltraFeedback%20Paper/purple)](https://arxiv.org/abs/2310.01377) - Advanced feedback collection |
| [![DeepSeek-R1 Local Fine-tuning Guide](https://badgen.net/badge/Blog/DeepSeek-R1%20Local%20Fine-tuning%20Guide/pink)](https://x.com/_avichawla/status/1884126766132011149) - Step-by-step local setup | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Self-Instruct](https://badgen.net/badge/Github%20Repository/Self-Instruct/cyan)](https://github.com/yizhongw/self-instruct) - Automated instruction generation | [![LIDA](https://badgen.net/badge/Github%20Repository/LIDA/cyan)](https://github.com/microsoft/LIDA) - Automatic dataset creation |
| [![Argilla](https://badgen.net/badge/Github%20Repository/Argilla/cyan)](https://github.com/argilla-io/argilla) - Data annotation platform | [![Stanford Alpaca Tools](https://badgen.net/badge/Github%20Repository/Stanford%20Alpaca%20Tools/cyan)](https://github.com/tatsu-lab/stanford_alpaca) - Dataset generation |

#### Popular Datasets
| Dataset | Description |
|----------|-------------|
| [![Anthropic Constitutional AI Dataset](https://badgen.net/badge/Hugging%20Face%20Dataset/Anthropic%20Constitutional%20AI%20Dataset/yellow)](https://huggingface.co/datasets/anthropic/constitutional-ai) | Aligned instruction dataset with safety considerations |
| [![OpenAssistant Conversations](https://badgen.net/badge/Hugging%20Face%20Dataset/OpenAssistant%20Conversations/yellow)](https://huggingface.co/datasets/OpenAssistant/oasst1) | High-quality conversational data |
| [![UltraChat](https://badgen.net/badge/Hugging%20Face%20Dataset/UltraChat/yellow)](https://huggingface.co/datasets/HuggingFaceH4/ultrachat) | Large-scale chat interactions |
| [![UltraFeedback](https://badgen.net/badge/Hugging%20Face%20Dataset/UltraFeedback/yellow)](https://huggingface.co/datasets/openbmb/UltraFeedback) | Comprehensive model evaluation data |
| [![Synthia-Coder-v1.5-I](https://badgen.net/badge/Hugging%20Face%20Dataset/Synthia-Coder-v1.5-I/yellow)](https://huggingface.co/datasets/migtissera/Synthia-Coder-v1.5-I) | 23.5K coding samples from Claude Opus |
| [![Synthetic Medical Conversations](https://badgen.net/badge/Hugging%20Face%20Dataset/Synthetic%20Medical%20Conversations/yellow)](https://huggingface.co/datasets/OnDeviceMedNotes/synthetic-medical-conversations-deepseek-v3) | Multilingual medical dialogues |
### Knowledge Distillation
- **Description**: Transfer expertise from large teacher models to smaller, efficient student models.
- **Concepts Covered**: `knowledge distillation`, `teacher-student`, `model compression`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Knowledge Distillation Explained](https://badgen.net/badge/Tutorial/Knowledge%20Distillation%20Explained/blue)](https://towardsdatascience.com/knowledge-distillation-simplified-ddc070724770) | [![DistilBERT Paper](https://badgen.net/badge/Paper/DistilBERT%20Paper/purple)](https://arxiv.org/abs/1910.01108) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Hugging Face Transformers](https://badgen.net/badge/Framework/Hugging%20Face%20Transformers/green)](https://huggingface.co/docs/transformers) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Knowledge Distillation Basics](https://badgen.net/badge/Notebook/Knowledge%20Distillation%20Basics/orange)](notebooks/knowledge_distillation_basics.ipynb) | Implement basic knowledge distillation |
| [![Student Model Training](https://badgen.net/badge/Notebook/Student%20Model%20Training/orange)](notebooks/student_model_training.ipynb) | Train efficient student models |

### Reasoning Models, Reinforcement Learning and Group Relative Policy Optimization (GRPO)
- **Description**: Explore models that enhance reasoning capabilities through chain-of-thought and GRPO-based training, focusing on efficient preference learning and resource-constrained environments.
- **Concepts Covered**: `chain-of-thought`, `reasoning`, `GRPO`, `preference learning`, `reward modeling`, `group-based advantage estimation`, `resource-efficient training`, `reasoning enhancement`, `reinforcement learning`, `long context scaling`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![DeepSeek R1 Reasoning Primer](https://badgen.net/badge/Blog/DeepSeek%20R1%20Reasoning%20Primer/pink)](https://aman.ai/primers/ai/deepseek-R1/) | [![DeepSeek R1 Theory Overview](https://badgen.net/badge/Video/DeepSeek%20R1%20Theory%20Overview/red)](https://www.youtube.com/watch?v=QdEuh2UVbu0) |
| [![DeepSeek GRPO Paper](https://badgen.net/badge/Paper/DeepSeek%20GRPO%20Paper/purple)](https://arxiv.org/pdf/2402.03300) | [![GRPO Explained by Yannic Kilcher](https://badgen.net/badge/Video/GRPO%20Explained%20by%20Yannic%20Kilcher/red)](https://youtube.com/watch?v=bAWV_yrqx4w) |
| [![DeepSeek R1 Reasoning Blog](https://badgen.net/badge/Blog/DeepSeek%20R1%20Reasoning%20Blog/pink)](https://unsloth.ai/blog/r1-reasoning) | [![Open-R1](https://badgen.net/badge/Blog/Open-R1/pink)](https://huggingface.co/blog/open-r1/update-1) |
| [![Kimi k1.5 Paper](https://badgen.net/badge/Paper/Kimi%20k1.5%20Paper/purple)](https://arxiv.org/abs/2401.12863) | [![AGIEntry Kimi Overview](https://badgen.net/badge/Website/AGIEntry%20Kimi%20Overview/blue)](https://agientry.com) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Unsloth](https://badgen.net/badge/Github%20Repository/Unsloth/cyan)](https://github.com/unslothai/unsloth) | [![TinyZero](https://badgen.net/badge/Github%20Repository/TinyZero/cyan)](https://github.com/Jiayi-Pan/TinyZero) |
| [![DeepSeek-R1 Training Framework](https://badgen.net/badge/Github%20Repository/DeepSeek-R1%20Training%20Framework/cyan)](https://github.com/deepseek-ai/DeepSeek-R1) | [![Kimi.ai](https://badgen.net/badge/Website/Kimi.ai/blue)](https://kimi.ai) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![GRPO Poetry Generation](https://badgen.net/badge/Colab%20Notebook/GRPO%20Poetry%20Generation/orange)](https://colab.research.google.com/drive/1Ty0ovsrpw8i-zJvDhlSAtBIVw3EZfHK5?usp=sharing) | Implement GRPO for poetry generation |
| [![Qwen 0.5B GRPO Implementation](https://badgen.net/badge/Colab%20Notebook/Qwen%200.5B%20GRPO%20Implementation/orange)](https://colab.research.google.com/drive/1bfhs1FMLW3FGa8ydvkOZyBNxLYOu0Hev?usp=sharing) | Train small models with GRPO |
| [![Phi-4 14B GRPO Training](https://badgen.net/badge/Colab%20Notebook/Phi-4%2014B%20GRPO%20Training/orange)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4_(14B)-GRPO.ipynb) | Scale GRPO to larger models |
| [![Llama 3.1 8B GRPO](https://badgen.net/badge/Colab%20Notebook/Llama%203.1%208B%20GRPO/orange)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb) | Advanced GRPO implementation |
### GRPO Datasets
- **Description**: Curated datasets for training and evaluating GRPO-based models, with focus on reasoning, poetry, and domain-specific tasks.
- **Concepts Covered**: `dataset curation`, `chain-of-thought patterns`, `reasoning verification`, `poetry generation`, `scientific problem-solving`, `data preprocessing`, `quality filtering`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Guide to Creating CoT Datasets](https://badgen.net/badge/Blog/Guide%20to%20Creating%20CoT%20Datasets/pink)](https://huggingface.co/blog/creating-chain-of-thought-datasets) | [![Scientific Dataset Curation Guide](https://badgen.net/badge/Github%20Repository/Scientific%20Dataset%20Curation%20Guide/cyan)](https://github.com/EricLu1/SCP-Guide) |
| [![Data Generation with R1 Models](https://badgen.net/badge/Github%20Repository/Data%20Generation%20with%20R1%20Models/cyan)](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/data_generation.md) | [![Verse Dataset Creation Tutorial](https://badgen.net/badge/Tutorial/Verse%20Dataset%20Creation%20Tutorial/blue)](https://github.com/PleIAs/verse-wikisource/blob/main/TUTORIAL.md) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Hugging Face Datasets](https://badgen.net/badge/Framework/Hugging%20Face%20Datasets/green)](https://huggingface.co/datasets) | [![Verse Dataset Tools](https://badgen.net/badge/Github%20Repository/Verse%20Dataset%20Tools/cyan)](https://github.com/PleIAs/verse-wikisource) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![CoT Dataset Creation](https://badgen.net/badge/Notebook/CoT%20Dataset%20Creation/orange)](notebooks/cot_dataset_creation.ipynb) | Build chain-of-thought datasets from scratch |
| [![Data Quality Assessment](https://badgen.net/badge/Notebook/Data%20Quality%20Assessment/orange)](notebooks/data_quality_assessment.ipynb) | Implement filtering and verification techniques |
| [![Poetry Dataset Generation](https://badgen.net/badge/Notebook/Poetry%20Dataset%20Generation/orange)](notebooks/poetry_dataset_generation.ipynb) | Create specialized poetry training data |

#### Popular Datasets
| Dataset | Description |
|----------|-------------|
| [![PleIAs Verse Wikisource](https://badgen.net/badge/Hugging%20Face%20Dataset/PleIAs%20Verse%20Wikisource/yellow)](https://huggingface.co/datasets/PleIAs/verse-wikisource) | 200,000 verses for poetry training |
| [![Bespoke-Stratos-17k](https://badgen.net/badge/Hugging%20Face%20Dataset/Bespoke-Stratos-17k/yellow)](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) | High-quality CoT with generation code |
| [![OpenThoughts-114k](https://badgen.net/badge/Hugging%20Face%20Dataset/OpenThoughts-114k/yellow)](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) | Comprehensive reasoning patterns distilled from R1 |
| [![Evalchemy Dataset](https://badgen.net/badge/Hugging%20Face%20Dataset/Evalchemy%20Dataset/yellow)](https://huggingface.co/datasets/evalchemy) | Complementary reasoning dataset |
| [![R1-Distill-SFT](https://badgen.net/badge/Hugging%20Face%20Dataset/R1-Distill-SFT/yellow)](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT) | 1.8M samples from DeepSeek-R1-32b |
| [![Sky-T1_data_17k](https://badgen.net/badge/Hugging%20Face%20Dataset/Sky-T1_data_17k/yellow)](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k) | 17k verified samples for coding, math, science |
| [![SCP-116K](https://badgen.net/badge/Hugging%20Face%20Dataset/SCP-116K/yellow)](https://huggingface.co/datasets/EricLu/SCP-116K) | Scientific problem-solving (Physics, Chemistry, Biology) |
| [![FineQwQ-142k](https://badgen.net/badge/Hugging%20Face%20Dataset/FineQwQ-142k/yellow)](https://huggingface.co/datasets/qingy2024/FineQwQ-142k) | Math, Coding, General reasoning |
| [![Dolphin-R1](https://badgen.net/badge/Hugging%20Face%20Dataset/Dolphin-R1/yellow)](https://huggingface.co/datasets/cognitivecomputations/dolphin-r1) | Combined R1 and Gemini 2 reasoning |
| [![Dolphin-R1-DeepSeek](https://badgen.net/badge/Hugging%20Face%20Dataset/Dolphin-R1-DeepSeek/yellow)](https://huggingface.co/datasets/mlabonne/dolphin-r1-deepseek) | DeepSeek-compatible format |