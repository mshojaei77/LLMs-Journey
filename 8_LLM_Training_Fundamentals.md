# Module 8: LLM Training Fundamentals

### Pretraining Objectives & Loss Functions
- **Description**: Understand the objectives and loss functions used to pretrain large language models.
- **Concepts Covered**: `pretraining`, `loss functions`, `masked language modeling`, `next-word prediction`, `resource optimization`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [Pretraining Objectives in NLP](https://ruder.io/nlp-imagenet/) | [Cross-Entropy Loss Explained](https://gombru.github.io/2018/05/23/cross_entropy_loss/) |
| [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories) | [Llama 3 Paper](https://arxiv.org/pdf/2407.21783) |
| [SmolGPT Training Guide](https://github.com/Om-Alve/smolGPT) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [TensorFlow](https://www.tensorflow.org/) | |

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
| [AdamW Optimizer](https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html) | |
| [Learning Rate Schedules](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [PyTorch Optim](https://pytorch.org/docs/stable/optim.html) | |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers) | |

### Hyperparameter Tuning & Experiment Management
- **Description**: Systematically tune hyperparameters and manage experiments for optimal model performance.
- **Concepts Covered**: `hyperparameter tuning`, `experiment tracking`, `grid search`, `random search`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [Hyperparameter Optimization Guide](https://wandb.ai/site/articles/hyperparameter-optimization-in-deep-learning) | |
| [Experiment Tracking with MLflow](https://www.mlflow.org/docs/latest/tracking.html) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [Weights & Biases](https://wandb.ai/) | |
| [MLflow](https://www.mlflow.org/) | |

### Training Stability & Convergence
- **Description**: Address challenges in training stability and ensure model convergence.
- **Concepts Covered**: `training stability`, `convergence`, `loss spikes`, `gradient clipping`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [Troubleshooting Deep Neural Networks](https://josh-tobin.com/troubleshooting-deep-neural-networks.html) | |
| [Stabilizing Training with Gradient Clipping](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [TensorFlow](https://www.tensorflow.org/) | |

### Synthetic Data Generation & Augmentation
- **Description**: Generate high-quality synthetic data to enhance training datasets and improve model performance.
- **Concepts Covered**: `synthetic data`, `data augmentation`, `self-instruct`, `bootstrapping`, `data distillation`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [Self-Instruct Paper](https://arxiv.org/abs/2212.10560) | [Data Distillation Techniques](https://arxiv.org/abs/2012.12242) |
| [Alpaca Approach](https://crfm.stanford.edu/2023/03/13/alpaca.html) | [WizardLM Self-Instruct Method](https://arxiv.org/abs/2304.12244) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [Self-Instruct](https://github.com/yizhongw/self-instruct) | [TextAugment](https://github.com/dsfsi/textaugment) |
| [LLM Dataset Processor](https://apify.com/dusan.vystrcil/llm-dataset-processor) | [NL-Augmenter](https://github.com/GEM-benchmark/NL-Augmenter) |
| [Synthetic Data Vault](https://sdv.dev/) | [GPT-3 Data Generation](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset) |
