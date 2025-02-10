# Module 15: Function Calling and AI Agents

### Function Calling Fundamentals
- **Description**: Learn how to enable LLMs to interact with external functions and APIs.
- **Concepts Covered**: `function calling`, `API integration`, `structured outputs`, `JSON schemas`, `database querying`, `parallel function calls`, `Pythonic function calling`
- **Learning Resources**:
  - [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
  - [LangChain Function Calling](https://python.langchain.com/docs/modules/model_io/output_parsers/structured)
  - [Querying Databases with Function Calling (2024)](https://arxiv.org/pdf/2502.00032) - Research paper on database querying using LLM function calling
  - [Dria Agent α Blog Post](https://huggingface.co/blog/andthattoo/dria-agent-a) - Insights into agentic LLM trained for Pythonic function calling
- **Tools**:
  - [OpenAI Function Calling API](https://platform.openai.com/docs/api-reference/chat/create#chat/create-functions)
  - [LangChain](https://python.langchain.com/)
  - [Gorilla Database Query Tool](https://github.com/weaviate/gorilla)
  - [Dria Agent Models](https://huggingface.co/driaforall) - 3B and 7B parameter variants for efficient function calling
- **Key Features**:
  - Parallel multi-function calls
  - Reasoning beyond JSON limitations
  - Edge device training capabilities
  - GPT-4 level performance on BFCL benchmark

### AI Agents & Autonomous Systems
- **Description**: Build autonomous AI agents and multi-agent systems that can plan and execute complex tasks.
- **Concepts Covered**: `autonomous agents`, `planning`, `task decomposition`, `tool use`, `multi-agent systems`, `agent communication`, `collaboration protocols`, `emergent behavior`, `layered memory`, `orchestration`, `multi-step workflows`
- **Learning Resources**:
  - [Building AI Agents with LangChain](https://python.langchain.com/docs/modules/agents/)
  - [AutoGPT Documentation](https://docs.agpt.co/)
  - [BabyAGI Paper](https://arxiv.org/abs/2305.12366)
  - [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
  - [Building AI Agents Newsletter](https://buildingaiagents.substack.com/) - Weekly expert insights
  - [Hugging Face Agents Course](https://huggingface.co/agents-course)
  - [Multi-Agent Systems Overview](https://arxiv.org/abs/2306.15330)
  - [Chain of Agents: LLMs Collaborating on Long-Context Tasks](https://research.google/chain-of-agents)
  - [Market Research Agent with CrewAI](https://github.com/shricastic/research-agent-crewai.git)
  - [Agentic RAG Tutorial](https://lorenzejay.dev/articles/practical-agentic-rag)

- **Tools & Frameworks**:
  - Agent Development:
    - [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
    - [AutoGen](https://github.com/microsoft/autogen)
    - [CrewAI](https://github.com/joaomdmoura/crewAI)
    - [MetaGPT](https://github.com/geekan/MetaGPT)
    - [OpenAI Swarm](https://github.com/openai/swarm/tree/main)
    - [AWS Multi-Agent Orchestrator](https://github.com/awslabs/multi-agent-orchestrator)
  - Workflow & Visualization:
    - [Langflow](https://github.com/logspace-ai/langflow)
    - [N8N](https://n8n.io/)
    - [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
    - [Smyth OS](https://smythos.com/)
    - [Rivet UI](https://rivet.ironcladapp.com/)
    - [Burr](https://github.com/DAGWorks-Inc/burr)
  - Development Platforms:
    - [Dify](https://dify.ai/)
    - [Microsoft Copilot Studio](https://copilot.microsoft.com/studio)
    - [Potpie AI](https://potpie.ai/)
    - [LangGraph](https://python.langchain.com/docs/langgraph)
    - [LangSmith](https://smith.langchain.com/)
  - Example Implementations:
    - [VLM Web Browser Agent](https://github.com/huggingface/smolagents/blob/main/examples/vlm_web_browser.py)

### Agent Evaluation & External Tools Integration
- **Description**: Implement evaluation frameworks and integrate external tools to enhance agent capabilities.
- **Concepts Covered**: `LLM-as-judge`, `quality metrics`, `evaluation frameworks`, `API integration`, `tool libraries`, `web services`, `data sources`
- **Learning Resources**:
  - [LangSmith Documentation](https://docs.smith.langchain.com/)
  - [Hugging Face Cookbook: Evaluating AI Search Engines](https://huggingface.co/learn/cookbook/llm_judge_evaluating_ai_search_engines_with_judges_library)
  - [LangChain Tools Documentation](https://python.langchain.com/docs/integrations/tools)
  - [Building Custom Tools Guide](https://python.langchain.com/docs/modules/agents/tools/custom_tools)
- **Tools**:
  - Evaluation:
    - [LangSmith](https://smith.langchain.com/)
    - [Judges Library](https://huggingface.co/docs/judges)
    - [OpenAI Evals](https://github.com/openai/evals)
  - Search & Data:
    - [SerpAPI](https://serpapi.com/)
    - [DuckDuckGo API](https://duckduckgo.com/api)
    - [Bing Web Search API](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api)
    - [Perplexity AI](https://www.perplexity.ai/)
    - [Exa AI](https://exa.ai/)
    - [Google Gemini](https://deepmind.google/technologies/gemini/)
  - Development:
    - [GitHub API](https://docs.github.com/en/rest)
    - [Shell Tools](https://python.langchain.com/docs/modules/agents/tools/shell)
    - [Python REPL](https://python.langchain.com/docs/modules/agents/tools/python)
    - [Pandas](https://pandas.pydata.org/)
    - [Requests](https://requests.readthedocs.io/)
    - [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/)

### AI Agents in different domains
- **Description**: Build AI agents for different domains, including medical, finance, and education.
- **Concepts Covered**: `medical`, `finance`, `education`, `agentic reasoning`, `multimodal integration`, `expert systems`
- **Learning Resources**:
  - [MedRAX Paper](https://arxiv.org/abs/2502.02673) - First medical reasoning agent for chest X-rays
  - [ChestAgentBench Dataset](https://huggingface.co/datasets/wanglab/chest-agent-bench) - Comprehensive medical agent benchmark with expert-curated clinical cases
  - [TradingGPT: Multi-Agent System with Layered Memory](https://arxiv.org/pdf/2309.03736) - Framework for enhanced financial trading using multi-agent LLMs with hierarchical memory
  - [AI Hedge Fund Implementation](https://github.com/virattt/ai-hedge-fund) - Multi-agent, multi-LLM system for financial analysis