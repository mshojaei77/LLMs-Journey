# Module 5: Modern LLM Architectures

### Encoder-Only Models (BERT)
- **Description**: Delve into bidirectional models used for language understanding.
- **Concepts Covered**: `BERT`, `bidirectional encoding`, `masked language modeling`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![BERT: Pre-training Deep Bidirectional Transformers](https://badgen.net/badge/Paper/BERT%3A%20Pre-training%20Deep%20Bidirectional%20Transformers/purple)](https://arxiv.org/abs/1810.04805) | [![Hugging Face BERT Guide](https://badgen.net/badge/Docs/Hugging%20Face%20BERT%20Guide/green)](https://huggingface.co/docs/transformers/model_doc/bert) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Hugging Face Transformers](https://badgen.net/badge/Framework/Hugging%20Face%20Transformers/green)](https://huggingface.co/docs/transformers) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![BERT Implementation](https://badgen.net/badge/Notebook/BERT%20Implementation/orange)](notebooks/bert_basics.ipynb) | Build a basic BERT model |
| [![BERT Fine-tuning](https://badgen.net/badge/Notebook/BERT%20Fine-tuning/orange)](notebooks/bert_finetune.ipynb) | Fine-tune BERT for classification |

### Decoder-Only Models (GPT)
- **Description**: Learn about autoregressive models optimized for text generation.
- **Concepts Covered**: `GPT`, `autoregressive modeling`, `next-word prediction`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Improving Language Understanding by Generative Pre-Training](https://badgen.net/badge/Paper/Improving%20Language%20Understanding%20by%20Generative%20Pre-Training/purple)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | [![Hugging Face GPT Guide](https://badgen.net/badge/Docs/Hugging%20Face%20GPT%20Guide/green)](https://huggingface.co/docs/transformers/model_doc/gpt2) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Hugging Face Transformers](https://badgen.net/badge/Framework/Hugging%20Face%20Transformers/green)](https://huggingface.co/docs/transformers) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![GPT Implementation](https://badgen.net/badge/Notebook/GPT%20Implementation/orange)](notebooks/gpt_basics.ipynb) | Build a basic GPT model |
| [![Text Generation](https://badgen.net/badge/Notebook/Text%20Generation/orange)](notebooks/text_generation.ipynb) | Generate text with GPT |

### Encoder-Decoder Models (T5)
- **Description**: Explore versatile models that combine encoder and decoder for sequence-to-sequence tasks.
- **Concepts Covered**: `T5`, `encoder-decoder`, `sequence-to-sequence`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://badgen.net/badge/Paper/Exploring%20the%20Limits%20of%20Transfer%20Learning/purple)](https://arxiv.org/abs/1910.10683) | [![Hugging Face T5 Guide](https://badgen.net/badge/Docs/Hugging%20Face%20T5%20Guide/green)](https://huggingface.co/docs/transformers/model_doc/t5) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Hugging Face Transformers](https://badgen.net/badge/Framework/Hugging%20Face%20Transformers/green)](https://huggingface.co/docs/transformers) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![T5 Implementation](https://badgen.net/badge/Notebook/T5%20Implementation/orange)](notebooks/t5_basics.ipynb) | Build a basic T5 model |
| [![Sequence Tasks](https://badgen.net/badge/Notebook/Sequence%20Tasks/orange)](notebooks/seq2seq_tasks.ipynb) | Apply T5 to various sequence tasks |

### Mixture of Experts (MoE) Models
- **Description**: Investigate models that scale efficiently by routing inputs to specialized expert networks.
- **Concepts Covered**: `MoE`, `sparse models`, `expert networks`, `switch transformers`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Switch Transformers Paper](https://badgen.net/badge/Paper/Switch%20Transformers%20Paper/purple)](https://arxiv.org/abs/2101.03961) | [![Mixture-of-Experts Explained](https://badgen.net/badge/Blog/Mixture-of-Experts%20Explained/pink)](https://huggingface.co/blog/moe) |
| [![UltraMem: A Memory-centric Alternative to Mixture-of-Experts](https://badgen.net/badge/Paper/UltraMem%3A%20A%20Memory-centric%20Alternative%20to%20MoE/purple)](https://arxiv.org/pdf/2411.12364) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![DeepSpeed MoE](https://badgen.net/badge/Framework/DeepSpeed%20MoE/green)](https://www.deepspeed.ai/tutorials/mixture-of-experts/) | [![Hugging Face Transformers](https://badgen.net/badge/Framework/Hugging%20Face%20Transformers/green)](https://huggingface.co/docs/transformers) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![MoE Implementation](https://badgen.net/badge/Notebook/MoE%20Implementation/orange)](notebooks/moe_basics.ipynb) | Build a basic MoE model |
| [![Expert Routing](https://badgen.net/badge/Notebook/Expert%20Routing/orange)](notebooks/expert_routing.ipynb) | Implement expert routing mechanisms |

### LLM Reasoning & Cognitive Architectures
- **Description**: Understand how LLMs perform different types of reasoning and their cognitive capabilities.
- **Concepts Covered**: `chain-of-thought`, `deductive reasoning`, `inductive reasoning`, `causal reasoning`, `multi-step reasoning`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Chain-of-Thought Paper](https://badgen.net/badge/Paper/Chain-of-Thought%20Paper/purple)](https://arxiv.org/abs/2201.11903) | [![Reasoning with Language Model Prompting: A Survey](https://badgen.net/badge/Paper/Reasoning%20with%20Language%20Model%20Prompting%3A%20A%20Survey/purple)](https://arxiv.org/abs/2212.09597) |
| [![Towards Reasoning in Large Language Models: A Survey](https://badgen.net/badge/Paper/Towards%20Reasoning%20in%20Large%20Language%20Models%3A%20A%20Survey/purple)](https://arxiv.org/abs/2212.10403) | [![A Visual Guide to Reasoning LLMs](https://badgen.net/badge/Blog/A%20Visual%20Guide%20to%20Reasoning%20LLMs/pink)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![LangChain ReAct](https://badgen.net/badge/Framework/LangChain%20ReAct/green)](https://python.langchain.com/docs/modules/agents/agent_types/react) | [![Tree of Thoughts](https://badgen.net/badge/Github%20Repository/Tree%20of%20Thoughts/cyan)](https://github.com/kyegomez/tree-of-thoughts) |
| [![Reflexion Framework](https://badgen.net/badge/Github%20Repository/Reflexion%20Framework/cyan)](https://github.com/noahshinn024/reflexion) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Chain-of-Thought Implementation](https://badgen.net/badge/Notebook/Chain-of-Thought%20Implementation/orange)](notebooks/cot_basics.ipynb) | Implement basic chain-of-thought reasoning |
| [![Multi-Step Reasoning](https://badgen.net/badge/Notebook/Multi-Step%20Reasoning/orange)](notebooks/multi_step_reasoning.ipynb) | Build complex reasoning chains |