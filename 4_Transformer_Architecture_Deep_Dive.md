# Module 4: Transformer Architecture Deep Dive

### The Attention Mechanism
- **Description**: Discover how attention enables models to focus on relevant parts of the input.
- **Concepts Covered**: `attention`, `softmax`, `context vectors`
- **Learning Resources**:
  - [![Transformers from Scratch](https://badgen.net/badge/Tutorial/Transformers%20from%20Scratch/blue)](https://brandonrohrer.com/transformers)
  - [![The Illustrated Transformer](https://badgen.net/badge/Blog/The%20Illustrated%20Transformer/cyan)](https://jalammar.github.io/illustrated-transformer/)
  - [![Attention? Attention! – Lilian Weng](https://badgen.net/badge/Blog/Attention%3F%20Attention%21/cyan)](https://lilianweng.github.io/posts/2018-06-24-attention/)
- **Tools**:
  - [![Hugging Face Transformers](https://badgen.net/badge/Docs/Hugging%20Face%20Transformers/green)](https://huggingface.co/docs/transformers)
  - [![BertViz](https://badgen.net/badge/Github%20Repository/BertViz/gray)](https://github.com/jessevig/bertviz)

### Self-Attention & Multi-Head Attention
- **Description**: Learn how self-attention allows tokens to weigh each other's importance and how multiple heads capture diverse relationships.
- **Concepts Covered**: `self-attention`, `multi-head attention`, `query-key-value`
- **Learning Resources**:
  - [![Self-Attention Explained](https://badgen.net/badge/Paper/Self-Attention%20Explained/purple)](https://arxiv.org/abs/1706.03762)
  - [![Multi-Head Attention Visualized](https://badgen.net/badge/Blog/Multi-Head%20Attention%20Visualized/cyan)](https://jalammar.github.io/illustrated-transformer/)
- **Tools**:
  - [![TensorFlow](https://badgen.net/badge/Framework/TensorFlow/green)](https://www.tensorflow.org/)

### Positional Encoding in Transformers
- **Description**: Add order information to token embeddings using positional encodings.
- **Concepts Covered**: `positional encoding`, `sinusoidal functions`, `learned embeddings`
- **Learning Resources**:
  - [![Positional Encoding Explorer](https://badgen.net/badge/Github%20Repository/Positional%20Encoding%20Explorer/gray)](https://github.com/jalammar/positional-encoding-explorer)
  - [![Rotary Embeddings Guide](https://badgen.net/badge/Blog/Rotary%20Embeddings%20Guide/cyan)](https://blog.eleuther.ai/rotary-embeddings/)
- **Tools**:
  - [![TensorFlow](https://badgen.net/badge/Framework/TensorFlow/green)](https://www.tensorflow.org/)

### Layer Normalization & Residual Connections
- **Description**: Improve training stability with normalization and skip connections.
- **Concepts Covered**: `layer normalization`, `residual connections`, `training stability`
- **Learning Resources**:
  - [![Layer Normalization Deep Dive](https://badgen.net/badge/Blog/Layer%20Normalization%20Deep%20Dive/cyan)](https://leimao.github.io/blog/Layer-Normalization/)
  - [![Residual Network Paper](https://badgen.net/badge/Paper/Residual%20Network%20Paper/purple)](https://arxiv.org/abs/1512.03385)
- **Tools**:
  - [![PyTorch LayerNorm](https://badgen.net/badge/Docs/PyTorch%20LayerNorm/green)](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
  - [![TensorFlow LayerNormalization](https://badgen.net/badge/Docs/TensorFlow%20LayerNormalization/green)](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)
