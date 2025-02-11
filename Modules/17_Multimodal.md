# Module 17: Multimodal Models

### Vision & Language
- **Description**: Work with models that combine visual and linguistic understanding.
- **Concepts Covered**: `vision-language models`, `image captioning`, `visual question answering`, `multimodal search`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![CLIP by OpenAI](https://badgen.net/badge/Website/CLIP_by_OpenAI/blue)](https://openai.com/research/clip) | [![Imagine while Reasoning in Space](https://badgen.net/badge/Paper/Imagine_while_Reasoning_in_Space/purple)](https://arxiv.org/pdf/2501.07542) |
| [![DeepSeek-VL Paper](https://badgen.net/badge/Paper/DeepSeek-VL_Paper/purple)](https://arxiv.org/pdf/2403.05525) | [![BLIP-3 Paper](https://badgen.net/badge/Paper/BLIP-3_Paper/purple)](https://arxiv.org/pdf/2408.08872) |
| [![Qwen2.5VL Blog Post](https://badgen.net/badge/Blog/Qwen2.5VL_Blog_Post/pink)](https://qwenlm.github.io/blog/qwen2.5-vl/) | |

https://research.nvidia.com/labs/dir/cosmos-tokenizer/ Cosmos Tokenizer: A suite of image and video tokenizers

| [![Spectral Image Tokenizer](https://badgen.net/badge/Paper/Spectral%20Image%20Tokenizer/purple)](http://arxiv.org/abs/2412.09607) | Novel approach using DWT coefficients for image tokenization |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![DeepSeek-VL](https://badgen.net/badge/Github Repository/DeepSeek-VL/cyan)](https://github.com/deepseek-ai/DeepSeek-V) | [![SmolVLM Demo & Models](https://badgen.net/badge/Hugging Face Space/SmolVLM_Demo_&_Models/yellow)](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM-256M-Demo) |
| [![Qwen2.5VL Models](https://badgen.net/badge/Hugging Face Model/Qwen2.5VL_Models_Collection/yellow)](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Vision-Language Integration](https://badgen.net/badge/Notebook/Vision-Language%20Integration/orange)](notebooks/vision_language_integration.ipynb) | Implementing basic vision-language tasks with DeepSeek-VL |
| [![OCR Pipeline](https://badgen.net/badge/Notebook/OCR%20Pipeline/orange)](notebooks/ocr_pipeline.ipynb) | Building an OCR system with Surya and DocTR |
| [![Multimodal Search](https://badgen.net/badge/Notebook/Multimodal%20Search/orange)](notebooks/multimodal_search.ipynb) | Creating a multimodal search system using CLIP |
| [![Handwritten Text Recognition](https://badgen.net/badge/Notebook/Handwritten%20Text%20Recognition/orange)](notebooks/handwritten_text.ipynb) | Implementing HTR-VT for handwritten text recognition |
| [![Fine-tuning Vision Models](https://badgen.net/badge/Notebook/Fine-tuning%20Vision%20Models/orange)](notebooks/vision_finetuning.ipynb) | Fine-tuning Qwen2.5VL for custom tasks |
| [![Code Generation](https://badgen.net/badge/Notebook/Code%20Generation/orange)](notebooks/code_generation.ipynb) | Building a code generation pipeline |
| [![Automated Repair](https://badgen.net/badge/Notebook/Automated%20Repair/orange)](notebooks/automated_repair.ipynb) | Implementing automated code repair |
| [![Debugging Assistant](https://badgen.net/badge/Notebook/Debugging%20Assistant/orange)](notebooks/debugging_assistant.ipynb) | Creating an LLM-powered debugging tool |


### Audio & Language
- **Description**: Work with models that combine audio and linguistic understanding.
- **Concepts Covered**: `audio-language models`, `speech recognition`, `audio classification`, `text-to-speech`, `voice cloning`, `audio event detection`, `music generation`, `audio enhancement`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Whisper Paper](https://badgen.net/badge/Paper/Whisper_Paper/purple)](https://arxiv.org/abs/2212.04356) | [![MusicGen Paper](https://badgen.net/badge/Paper/MusicGen_Paper/purple)](https://arxiv.org/abs/2306.05284) |
| [![Bark Paper](https://badgen.net/badge/Paper/Bark_Paper/purple)](https://arxiv.org/abs/2307.07146) | [![AudioCraft Blog](https://badgen.net/badge/Blog/AudioCraft_Blog/pink)](https://ai.meta.com/blog/audiocraft-musicgen-audiogen-encodec-generative-ai-audio/) |
| [![Stable Audio Paper](https://badgen.net/badge/Paper/Stable_Audio_Paper/purple)](https://arxiv.org/abs/2401.04577) | [![Voice Cloning Ethics](https://badgen.net/badge/Website/Voice_Cloning_Ethics/blue)](https://elevenlabs.io/ethics) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Whisper](https://badgen.net/badge/Github Repository/Whisper/cyan)](https://github.com/openai/whisper) | [![Coqui TTS](https://badgen.net/badge/Github Repository/Coqui_TTS/cyan)](https://github.com/coqui-ai/TTS) |
| [![Bark](https://badgen.net/badge/Github Repository/Bark/cyan)](https://github.com/suno-ai/bark) | [![AudioCraft](https://badgen.net/badge/Github Repository/AudioCraft/cyan)](https://github.com/facebookresearch/audiocraft) |
| [![Stable Audio](https://badgen.net/badge/Website/Stable_Audio/blue)](https://www.stability.ai/stable-audio) | [![ElevenLabs](https://badgen.net/badge/Website/ElevenLabs/blue)](https://elevenlabs.io/) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Speech Recognition](https://badgen.net/badge/Notebook/Speech%20Recognition/orange)](notebooks/speech_recognition.ipynb) | Implementing Whisper for transcription |
| [![Text to Speech](https://badgen.net/badge/Notebook/Text%20to%20Speech/orange)](notebooks/text_to_speech.ipynb) | Building TTS systems with Bark |
| [![Voice Cloning](https://badgen.net/badge/Notebook/Voice%20Cloning/orange)](notebooks/voice_cloning.ipynb) | Creating voice cloning applications |
| [![Music Generation](https://badgen.net/badge/Notebook/Music%20Generation/orange)](notebooks/music_generation.ipynb) | Generating music with MusicGen |
| [![Audio Enhancement](https://badgen.net/badge/Notebook/Audio%20Enhancement/orange)](notebooks/audio_enhancement.ipynb) | Implementing audio enhancement techniques |
| [![Event Detection](https://badgen.net/badge/Notebook/Event%20Detection/orange)](notebooks/event_detection.ipynb) | Building audio event detection systems |