# Module 4: Transformer Architecture Deep Dive

### The Attention Mechanism
Discover how attention enables models to focus on relevant parts of the input.

**Key Concepts**: `attention`, `softmax`, `context vectors`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Transformers from Scratch](https://badgen.net/badge/Tutorial/Transformers%20from%20Scratch/blue)](https://brandonrohrer.com/transformers) | [![Attention? Attention! – Lilian Weng](https://badgen.net/badge/Blog/Attention%3F%20Attention%21/pink)](https://lilianweng.github.io/posts/2018-06-24-attention/) |
| [![The Illustrated Transformer](https://badgen.net/badge/Blog/The%20Illustrated%20Transformer/pink)](https://jalammar.github.io/illustrated-transformer/) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Hugging Face Transformers](https://badgen.net/badge/Docs/Hugging%20Face%20Transformers/green)](https://huggingface.co/docs/transformers) | [![BertViz](https://badgen.net/badge/Github%20Repository/BertViz/cyan)](https://github.com/jessevig/bertviz) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [Attention Basics](notebooks/attention_basics.ipynb) | Build a basic attention mechanism from scratch |
| [Attention Visualization](notebooks/attention_viz.ipynb) | Visualize attention patterns in transformer models |

### Self-Attention & Multi-Head Attention
Understand how self-attention enables tokens to interact and how multiple attention heads capture different relationship patterns.

**Key Concepts**: `self-attention`, `multi-head attention`, `query-key-value`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Self-Attention Explained](https://badgen.net/badge/Paper/Self-Attention%20Explained/purple)](https://arxiv.org/abs/1706.03762) | [![Multi-Head Attention Visualized](https://badgen.net/badge/Blog/Multi-Head%20Attention%20Visualized/pink)](https://jalammar.github.io/illustrated-transformer/) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![TensorFlow](https://badgen.net/badge/Framework/TensorFlow/green)](https://www.tensorflow.org/) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [Self-Attention Implementation](notebooks/self_attention.ipynb) | Build self-attention from scratch |
| [Multi-Head Analysis](notebooks/multi_head.ipynb) | Analyze different attention heads |

### Positional Encoding in Transformers
Add sequential information to token embeddings through positional encodings.

**Key Concepts**: `positional encoding`, `sinusoidal functions`, `learned embeddings`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Positional Encoding Explorer](https://badgen.net/badge/Github%20Repository/Positional%20Encoding%20Explorer/cyan)](https://github.com/jalammar/positional-encoding-explorer) | [![Rotary Embeddings Guide](https://badgen.net/badge/Blog/Rotary%20Embeddings%20Guide/pink)](https://blog.eleuther.ai/rotary-embeddings/) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![TensorFlow](https://badgen.net/badge/Framework/TensorFlow/green)](https://www.tensorflow.org/) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [Positional Encoding Basics](notebooks/pos_encoding.ipynb) | Implement basic positional encodings |
| [RoPE Implementation](notebooks/rope.ipynb) | Build rotary position embeddings |

### Layer Normalization & Residual Connections
Enhance training stability through normalization techniques and skip connections.

**Key Concepts**: `layer normalization`, `residual connections`, `training stability`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Layer Normalization Deep Dive](https://badgen.net/badge/Blog/Layer%20Normalization%20Deep%20Dive/pink)](https://leimao.github.io/blog/Layer-Normalization/) | [![Residual Network Paper](https://badgen.net/badge/Paper/Residual%20Network%20Paper/purple)](https://arxiv.org/abs/1512.03385) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![PyTorch LayerNorm](https://badgen.net/badge/Docs/PyTorch%20LayerNorm/green)](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) | [![TensorFlow LayerNormalization](https://badgen.net/badge/Docs/TensorFlow%20LayerNormalization/green)](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [LayerNorm Implementation](notebooks/layer_norm.ipynb) | Build layer normalization from scratch |
| [ResNet Connections](notebooks/residual.ipynb) | Add residual connections to networks |
