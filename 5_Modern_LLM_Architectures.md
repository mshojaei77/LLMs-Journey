# Module 5: Modern LLM Architectures

### Encoder-Only Models (BERT)
- **Description**: Delve into bidirectional models used for language understanding.
- **Concepts Covered**: `BERT`, `bidirectional encoding`, `masked language modeling`
- **Learning Resources**:
  - [BERT: Pre-training Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
  - [Hugging Face BERT Guide](https://huggingface.co/docs/transformers/model_doc/bert)
- **Tools**:
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers)

### Decoder-Only Models (GPT)
- **Description**: Learn about autoregressive models optimized for text generation.
- **Concepts Covered**: `GPT`, `autoregressive modeling`, `next-word prediction`
- **Learning Resources**:
  - [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  - [Hugging Face GPT Guide](https://huggingface.co/docs/transformers/model_doc/gpt2)
- **Tools**:
    - [Hugging Face Transformers](https://huggingface.co/docs/transformers)

### Encoder-Decoder Models (T5)
- **Description**: Explore versatile models that combine encoder and decoder for sequence-to-sequence tasks.
- **Concepts Covered**: `T5`, `encoder-decoder`, `sequence-to-sequence`
- **Learning Resources**:
  - [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
  - [Hugging Face T5 Guide](https://huggingface.co/docs/transformers/model_doc/t5)
- **Tools**:
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers)

### Mixture of Experts (MoE) Models
- **Description**: Investigate models that scale efficiently by routing inputs to specialized expert networks.
- **Concepts Covered**: `MoE`, `sparse models`, `expert networks`, `switch transformers`
- **Learning Resources**:
  - [Switch Transformers Paper](https://arxiv.org/abs/2101.03961)
  - [Mixture-of-Experts Explained](https://huggingface.co/blog/moe)
  - [UltraMem: A Memory-centric Alternative to Mixture-of-Experts](https://arxiv.org/pdf/2411.12364) - Novel sparse model architecture with:

- **Tools**:
  - [DeepSpeed MoE](https://www.deepspeed.ai/tutorials/mixture-of-experts/)
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers)

### LLM Reasoning & Cognitive Architectures
- **Description**: Understand how LLMs perform different types of reasoning and their cognitive capabilities.
- **Concepts Covered**: `chain-of-thought`, `deductive reasoning`, `inductive reasoning`, `causal reasoning`, `multi-step reasoning`
- **Learning Resources**:
  - [Reasoning with Language Model Prompting: A Survey](https://arxiv.org/abs/2212.09597)
  - [Chain-of-Thought Paper](https://arxiv.org/abs/2201.11903)
  - [Towards Reasoning in Large Language Models: A Survey](https://arxiv.org/abs/2212.10403)
  - [A Visual Guide to Reasoning LLMs](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms) - Comprehensive visual exploration of DeepSeek-R1, train-time compute paradigms, and reasoning techniques
- **Tools**:
  - [LangChain ReAct](https://python.langchain.com/docs/modules/agents/agent_types/react)
  - [Tree of Thoughts](https://github.com/kyegomez/tree-of-thoughts)
  - [Reflexion Framework](https://github.com/noahshinn024/reflexion)