# Module 8: LLM Training Fundamentals

### Pretraining Objectives & Loss Functions
- **Description**: Understand the objectives and loss functions used to pretrain large language models.
- **Concepts Covered**: `pretraining`, `loss functions`, `masked language modeling`, `next-word prediction`, `resource optimization`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Pretraining Objectives in NLP](https://badgen.net/badge/Blog/Pretraining%20Objectives%20in%20NLP/pink)](https://ruder.io/nlp-imagenet/) | [![Cross-Entropy Loss Explained](https://badgen.net/badge/Blog/Cross-Entropy%20Loss%20Explained/pink)](https://gombru.github.io/2018/05/23/cross_entropy_loss/) |
| [![TinyStories Dataset](https://badgen.net/badge/Hugging%20Face%20Dataset/TinyStories/yellow)](https://huggingface.co/datasets/roneneldan/TinyStories) | [![Llama 3 Paper](https://badgen.net/badge/Paper/Llama%203/purple)](https://arxiv.org/pdf/2407.21783) |
| [![SmolGPT Training Guide](https://badgen.net/badge/Github%20Repository/SmolGPT/cyan)](https://github.com/Om-Alve/smolGPT) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![TensorFlow](https://badgen.net/badge/Framework/TensorFlow/green)](https://www.tensorflow.org/) | |

#### Cost & Resource Considerations:
- Training a 27.5M parameter model on 4B tokens: ~$13 and 18.5 hours
- Experimentation and optimization costs: ~$50
- Efficient architecture choices can significantly reduce training costs

### Optimization Strategies for LLMs
- **Description**: Explore optimizers and learning rate schedules tailored for LLM training.
- **Concepts Covered**: `optimization`, `AdamW`, `learning rate schedules`, `warmup`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![AdamW Optimizer](https://badgen.net/badge/Blog/AdamW%20Optimizer/pink)](https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html) | |
| [![Learning Rate Schedules](https://badgen.net/badge/Docs/Learning%20Rate%20Schedules/green)](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![PyTorch Optim](https://badgen.net/badge/Framework/PyTorch%20Optim/green)](https://pytorch.org/docs/stable/optim.html) | |
| [![Hugging Face Transformers](https://badgen.net/badge/Framework/Hugging%20Face%20Transformers/green)](https://huggingface.co/docs/transformers) | |

### Hyperparameter Tuning & Experiment Management
- **Description**: Systematically tune hyperparameters and manage experiments for optimal model performance.
- **Concepts Covered**: `hyperparameter tuning`, `experiment tracking`, `grid search`, `random search`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Hyperparameter Optimization Guide](https://badgen.net/badge/Tutorial/Hyperparameter%20Optimization%20Guide/blue)](https://wandb.ai/site/articles/hyperparameter-optimization-in-deep-learning) | |
| [![Experiment Tracking with MLflow](https://badgen.net/badge/Tutorial/Experiment%20Tracking%20with%20MLflow/blue)](https://www.mlflow.org/docs/latest/tracking.html) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Weights & Biases](https://badgen.net/badge/Framework/Weights%20%26%20Biases/green)](https://wandb.ai/) | |
| [![MLflow](https://badgen.net/badge/Framework/MLflow/green)](https://www.mlflow.org/) | |

### Training Stability & Convergence
- **Description**: Address challenges in training stability and ensure model convergence.
- **Concepts Covered**: `training stability`, `convergence`, `loss spikes`, `gradient clipping`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Troubleshooting Deep Neural Networks](https://badgen.net/badge/Tutorial/Troubleshooting%20Deep%20Neural%20Networks/blue)](https://josh-tobin.com/troubleshooting-deep-neural-networks.html) | |
| [![Stabilizing Training with Gradient Clipping](https://badgen.net/badge/Docs/Stabilizing%20Training%20with%20Gradient%20Clipping/green)](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![TensorFlow](https://badgen.net/badge/Framework/TensorFlow/green)](https://www.tensorflow.org/) | |

### Synthetic Data Generation & Augmentation
- **Description**: Generate high-quality synthetic data to enhance training datasets and improve model performance.
- **Concepts Covered**: `synthetic data`, `data augmentation`, `self-instruct`, `bootstrapping`, `data distillation`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Self-Instruct Paper](https://badgen.net/badge/Paper/Self-Instruct/purple)](https://arxiv.org/abs/2212.10560) | [![Data Distillation Techniques](https://badgen.net/badge/Paper/Data%20Distillation%20Techniques/purple)](https://arxiv.org/abs/2012.12242) |
| [![Alpaca Approach](https://badgen.net/badge/Blog/Alpaca%20Approach/pink)](https://crfm.stanford.edu/2023/03/13/alpaca.html) | [![WizardLM Self-Instruct Method](https://badgen.net/badge/Paper/WizardLM%20Self-Instruct%20Method/purple)](https://arxiv.org/abs/2304.12244) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Self-Instruct](https://badgen.net/badge/Github%20Repository/Self-Instruct/cyan)](https://github.com/yizhongw/self-instruct) | [![TextAugment](https://badgen.net/badge/Github%20Repository/TextAugment/cyan)](https://github.com/dsfsi/textaugment) |
| [![LLM Dataset Processor](https://badgen.net/badge/API%20Provider/LLM%20Dataset%20Processor/blue)](https://apify.com/dusan.vystrcil/llm-dataset-processor) | [![NL-Augmenter](https://badgen.net/badge/Github%20Repository/NL-Augmenter/cyan)](https://github.com/GEM-benchmark/NL-Augmenter) |
| [![Synthetic Data Vault](https://badgen.net/badge/Framework/Synthetic%20Data%20Vault/green)](https://sdv.dev/) | [![GPT-3 Data Generation](https://badgen.net/badge/Docs/GPT-3%20Data%20Generation/green)](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset) |
