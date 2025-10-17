<h1 align = "center">Large Language Models for Personalized Alignment</h1>

<p align="center">
  <img src="https://img.shields.io/github/stars/Tomsawyerhu/Awesome-Personalized-Alignment?color=yellow&label=Stars">
</p>

A collection of academic publications and methodologies on the personalized alignment of Large Language Models, covering training-time methods (fine-tuning, embedding learning, learning from human feedback, reinforcement learning) and test-time methods (guided decoding,prompt-based, RAG, agent, memory)

We welcome all researchers to contribute to this repository and further contribute to the knowledge of the Large Language Models with Personalized Alignment field. Please feel free to contact us if you have any related references by Github issue or pull request.

## 📖 Contents

- [📄 Related Surveys](#-related-surveys)
- [💡 Test-time Personalized Alignment](#-test-time-personalized-alignment)
  - [🔤 Guided Decoding](#-guided-decoding)
  - [💬 Prompt-based](#-prompt-based)
  - [🔎 RAG](#-rag)
  - [🤖 Agent](#-agent)
  - [🧠 Memory](#-memory)
- [⚙️ Training-time Personalized Alignment](#-training-time-personalized-alignment)
  - [🔧 Fine-tune](#-fine-tune)
  - [📌 Embedding Learning](#-embedding-learning)
  - [👥 Learning From Human Feedback](#-learning-from-human-feedback)
  - [🔄 Reinforcement Learning](#-reinforcement-learning)
 

## 📄 Related Surveys

| Title | Date | Link(s) |
|-------|------|---------|
| A Survey on Personalized and Pluralistic Preference Alignment in Large Language Models | 2025/4 | [arXiv](https://arxiv.org/abs/2504.07070v1) |
| A Survey of Personalization: From RAG to Agent | 2025/4 | [arXiv](https://arxiv.org/abs/2504.10147) | [GitHub](https://github.com/Applied-Machine-Learning-Lab/Awesome-Personalized-RAG-Agent) |
| A Survey on Personalized Alignment -- The Missing Piece for Large Language Models in Real-World Applications | 2025/3 | [arXiv](https://arxiv.org/abs/2503.17003) |
| Personalized Generation In Large Model Era: A Survey | 2025/3 | [arXiv](https://arxiv.org/abs/2503.02614) |
| A Survey of Personalized Large Language Models: Progress and Future Directions | 2025/2 | [arXiv](https://arxiv.org/abs/2502.11528) , [GitHub](https://github.com/JiahongLiu21/Awesome-Personalized-Large-Language-Models) |
| A Survey on Alignment for Large Language Model Agents | 2025/2 | [OpenReview](https://openreview.net/pdf?id=gkxt5kZS84) |
| Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization | 2024/12 | [arXiv](https://arxiv.org/abs/2406.01171) , [GitHub](https://github.com/MiuLab/PersonaLLM-Survey) |
| Personalized Multimodal Large Language Models: A Survey | 2024/12 | [arXiv](https://arxiv.org/abs/2412.02142) |
| Personalization of Large Language Models: A Survey | 2024/11 | [arXiv](https://arxiv.org/abs/2411.00027) |
| When large language models meet personalization: perspectives of challenges and opportunities | 2024/10 | [arXiv](https://arxiv.org/abs/2307.16376) |
| The Multilingual Alignment Prism: Aligning Global and Local Preferences to Reduce Harm | 2024/07 | [arXiv](https://arxiv.org/abs/2406.18682) |
| Recent Trends in Personalized Dialogue Generation: A Review of Datasets, Methodologies, and Evaluations | 2024/05 | [arXiv](https://arxiv.org/abs/2405.17974) |
| The benefits, risks and bounds of personalizing the alignment of large language models to individuals | 2024/04 | [arXiv](https://arxiv.org/abs/2303.05453) |
| From Persona to Personalization: A Survey on Role-Playing Language Agents | 2024/04 | [arXiv](https://arxiv.org/abs/2404.18231) |
| A Survey on the Memory Mechanism of Large Language Model based Agents | 2024/04 | [arXiv](https://arxiv.org/abs/2404.13501), [Github](https://github.com/nuster1128/LLM_Agent_Memory_Survey) |
| Position: A Roadmap to Pluralistic Alignment | 2024/02 | [arXiv](https://arxiv.org/abs/2402.05070) , [GitHub](https://github.com/jfisher52/AI_Pluralistic_Alignment) |

## 💡 Test-time Personalized Alignment
#### 🔤 Guided Decoding

| Paper | Abbr. | Year | Source | Tag | PDF | 
|----------|------|------|------|------|--------|
| [Args: Alignment as reward-guided search](http://arxiv.org/abs/1912.00180v1) | Args | 2024 | arxiv | reward-guided decoding | [PDF](http://arxiv.org/pdf/1912.00180v1.pdf) | 
| [Reward-augmented decoding: Efficient controlled text generation with a unidirectional reward model](http://arxiv.org/abs/2310.09520v4) | RAD | 2023 | arxiv | reward-guided decoding | [PDF](http://arxiv.org/pdf/2310.09520v4.pdf) | 
| [Cascade reward sampling for efficient decoding-time alignment](http://arxiv.org/abs/2406.16306v3) | CARDS | 2024 | arxiv | reward-guided decoding | [PDF](http://arxiv.org/pdf/2406.16306v3.pdf) | 
| [Deal: Decoding-time alignment for large language models](http://arxiv.org/abs/2410.01079v1) | DeAL | 2024 | arxiv | reward-guided decoding | [PDF](http://arxiv.org/pdf/2410.01079v1.pdf) | 
| [Genarm: Reward guided generation with autoregressive reward model for test-time alignment](http://arxiv.org/abs/2410.08193v5) | GenARM | 2024 | arxiv | reward-guided decoding | [PDF](http://arxiv.org/pdf/2410.08193v5.pdf) | 
| [Pad: Personalized alignment at decoding-time](http://arxiv.org/abs/2211.04198v1) | PAD | 2024 | arxiv | reward-guided decoding | [PDF](http://arxiv.org/pdf/2211.04198v1.pdf) | 
| [Controlled decoding from language models](http://arxiv.org/abs/2212.10938v1) | CD | 2023 | arxiv | reward-guided decoding | [PDF](http://arxiv.org/pdf/2212.10938v1.pdf) | 
| [Don’t throw away your value model! generating more preferable text with value-guided monte-carlo tree search decoding](http://arxiv.org/abs/1008.1442v1) | PPO-MCTS | 2023 | arxiv | reward-guided decoding | [PDF](http://arxiv.org/pdf/1008.1442v1.pdf) | 
| [Inference-time language model alignment via integrated value guidance](http://arxiv.org/abs/2409.17819v1) | IVG | 2024 | arxiv | reward-guided decoding | [PDF](http://arxiv.org/pdf/2409.17819v1.pdf) | 
| [Value augmented sampling for language model alignment and personalization](http://arxiv.org/abs/2405.06639v1) | VAS | 2024 | arxiv | reward-guided decoding | [PDF](http://arxiv.org/pdf/2405.06639v1.pdf) | 
| [Language model personalization via reward factorization](http://arxiv.org/abs/2503.06358v1) | | 2025 | arxiv | reward-guided decoding | [PDF](http://arxiv.org/pdf/2503.06358v1.pdf) | 
| [Persona-judge: Personalized alignment of large language models via token-level self-judgment](http://arxiv.org/abs/2403.17141v3) | Persona-judge | 2025 | arxiv | reward-guided decoding | [PDF](http://arxiv.org/pdf/2403.17141v3.pdf) | 
| [MAVIS: Multi-Objective Alignment via Value-Guided Inference-Time Search](http://arxiv.org/abs/2208.11125v1) | MAVIS | 2025 | arxiv | reward-guided decoding | [PDF](http://arxiv.org/pdf/2208.11125v1.pdf) | 
| [PITA: Preference-Guided Inference-Time Alignment for LLM Post-Training](http://arxiv.org/abs/2507.07725v1) | PITA | 2025 | arxiv | reward-guided decoding | [PDF](http://arxiv.org/pdf/2507.07725v1.pdf) | 
| [Parm: Multi-objective test-time alignment via preference-aware autoregressive reward model](http://arxiv.org/abs/2410.08193v5) | Parm | 2025 | arxiv | reward-guided decoding | [PDF](http://arxiv.org/pdf/2410.08193v5.pdf) | 
| [Search-Based Interaction For Conversation Recommendation via Generative Reward Model Based Simulated User](http://arxiv.org/abs/2504.20458v1) | GRSU | 2025 | arxiv | reward-guided decoding | [PDF](http://arxiv.org/pdf/2504.20458v1.pdf) | 

#### 💬 Prompt-based

| Paper | Abbr. | Year | Source | Tag | PDF | 
|----------|------|------|------|------|--------|
| [Evaluating and inducing personality in pre-trained language models](http://arxiv.org/abs/2508.06149v1) | P2 | 2023 | NIPS | Direct Prompting | [PDF](http://arxiv.org/pdf/2508.06149v1.pdf) | 
| [Evaluating character understanding of large language models via character profiling from fictional works.](http://arxiv.org/abs/2404.12726v3) | Character Profiling | 2024 | arxiv | Direct Prompting | [PDF](http://arxiv.org/pdf/2404.12726v3.pdf) | 
| [Whose opinions do language models reflect?](http://arxiv.org/abs/2303.17548v1) | OpinionQA | 2023 | PMLR | Direct Prompting | [PDF](http://arxiv.org/pdf/2303.17548v1.pdf) | 
| [Do llms understand user preferences? evaluating llms on user rating prediction](http://arxiv.org/abs/2305.06474v1) | | 2023 | arxiv | Direct Prompting | [PDF](http://arxiv.org/pdf/2305.06474v1.pdf) | 
| [Is chatgpt a good recommender? a preliminary study](http://arxiv.org/abs/2307.03952v3) | | 2023 | arxiv | Direct Prompting | [PDF](http://arxiv.org/pdf/2307.03952v3.pdf) | 
| [Cue-CoT: Chain-of-thought prompting for responding to in-depth dialogue questions with LLMs](http://arxiv.org/abs/2305.11792v2) | Cue-CoT | 2023 | arxiv | Direct Prompting | [PDF](http://arxiv.org/pdf/2305.11792v2.pdf) | 
| [Tuning-Free Personalized Alignment via Trial-Error-Explain In-Context Learning](http://arxiv.org/abs/2402.10207v6) | TICL | 2025 | arxiv | Direct Prompting | [PDF](http://arxiv.org/pdf/2402.10207v6.pdf) | 
| [ALIGN: Prompt-based Attribute Alignment for Reliable, Responsible, and Personalized LLM-based Decision-Making](http://arxiv.org/abs/2507.09037v1) | ALIGN | 2025 | arxiv | Direct Prompting | [PDF](http://arxiv.org/pdf/2507.09037v1.pdf) | 
| [Guided profile generation improves personalization with llms](http://arxiv.org/abs/2409.13093v1) | GPG | 2024 | arxiv | Profile-Augmented Prompting | [PDF](http://arxiv.org/pdf/2409.13093v1.pdf) | 
| [Integrating summarization and retrieval for enhanced personalization via large language models](http://arxiv.org/abs/2310.20081v1) | | 2023 | arxiv | Profile-Augmented Prompting | [PDF](http://arxiv.org/pdf/2310.20081v1.pdf) | 
| [Once:Boostingcontent-basedrecommendationwithbothopen-andclosed-source large language models](http://arxiv.org/abs/2306.07377v1) | ONCE | 2024 | WSDM | Profile-Augmented Prompting | [PDF](http://arxiv.org/pdf/2306.07377v1.pdf) | 
| [LLMTreeRec: Unleashing the Power of Large Language Models for Cold-Start Recommendations](http://arxiv.org/abs/2205.13795v1) | LLMTreeRec | 2025 | COLING | Profile-Augmented Prompting | [PDF](http://arxiv.org/pdf/2205.13795v1.pdf) | 
| [Towards open-world recommendation with knowledge augmentation from large language models](http://arxiv.org/abs/2306.10933v4) | KAR | 2024 | RecSys | Profile-Augmented Prompting | [PDF](http://arxiv.org/pdf/2306.10933v4.pdf) | 
| [Matryoshka: Learning to Drive Black-Box LLMs with LLMs](http://arxiv.org/abs/2310.01957v2) | Matryoshka | 2024 | arxiv | Profile-Augmented Prompting | [PDF](http://arxiv.org/pdf/2310.01957v2.pdf) | 
| [Few-shot personalization of llms with mis-aligned re- sponses](http://arxiv.org/abs/1003.3309v1) | FERMI | 2024 | arxiv | Profile-Augmented Prompting | [PDF](http://arxiv.org/pdf/1003.3309v1.pdf) | 
| [Learning to rewrite prompts for personalized text generation](http://arxiv.org/abs/2310.00152v2) | | 2024 | WWW | Personalized-Prompt Prompting | [PDF](http://arxiv.org/pdf/2310.00152v2.pdf) | 
| [Recgpt: Generative personalized prompts for sequential recommendation via chatgpt training paradigm](http://arxiv.org/abs/2404.08675v1) | RecGPT | 2024 | arxiv | Personalized-Prompt Prompting | [PDF](http://arxiv.org/pdf/2404.08675v1.pdf) | 
| [Personalized prompt learning for explainable recommendation](http://arxiv.org/abs/2205.09666v3) | PEPLER-D | 2023 | TOIS | Personalized-Prompt Prompting | [PDF](http://arxiv.org/pdf/2205.09666v3.pdf) | 
| [Graph-enhanced prompt learning for personalized review generation](http://arxiv.org/abs/2505.22447v1) | GRAPA | 2024 | DSE | Personalized-Prompt Prompting | [PDF](http://arxiv.org/pdf/2505.22447v1.pdf) | 
| [Unlocking the potential of prompt-tuning in bridging generalized and personalized federated learning](http://arxiv.org/abs/2310.18285v4) | SGPT | 2024 | CVPR | Personalized-Prompt Prompting | [PDF](http://arxiv.org/pdf/2310.18285v4.pdf) | 
| [Personalized federated continual learning via multi-granularity prompt](http://arxiv.org/abs/2407.00113v1) | PFCL | 2024 | KDD | Personalized-Prompt Prompting | [PDF](http://arxiv.org/pdf/2407.00113v1.pdf) | 
| [The unlocking spell on base llms: Rethink- ing alignment via in-context learning](http://arxiv.org/abs/2312.01552v1) | URIAL | 2023 | ICLR | Personalized-Prompt Prompting | [PDF](http://arxiv.org/pdf/2312.01552v1.pdf) | 
| [Black-box prompt optimization: Aligning large language models without model training](http://arxiv.org/abs/2307.12980v1) | BPO | 2023 | arxiv | Personalized-Prompt Prompting | [PDF](http://arxiv.org/pdf/2307.12980v1.pdf) | 
| [Reinforced prompt personalization for recommendation with large language models](http://arxiv.org/abs/2407.17115v2) | RPP | 2025 | TOIS | Personalized-Prompt Prompting | [PDF](http://arxiv.org/pdf/2407.17115v2.pdf) | 
| [Rain: Your language models can align themselves without finetuning](http://arxiv.org/abs/2309.07124v2) | RAIN | 2023 | arxiv | Prompt-based refine | [PDF](http://arxiv.org/pdf/2309.07124v2.pdf) | 
| [Learning to rewrite prompts for personalized text generation](http://arxiv.org/abs/2310.00152v2) | Rewrite Prompt | 2024 | WWW | Prompt-based refine | [PDF](http://arxiv.org/pdf/2310.00152v2.pdf) | 

#### 🔎 RAG

| Paper | Abbr. | Year | Source | Tag | PDF | 
|----------|------|------|------|------|--------|
| [Lamp: When large language models meet personalization](http://arxiv.org/abs/2304.11406v4) | Lamp | 2023 | arxiv | RAG | [PDF](http://arxiv.org/pdf/2304.11406v4.pdf) | 
| [Crafting Personalized Agents through Retrieval-Augmented Generation on Editable Memory Graphs](http://arxiv.org/abs/2409.19401v1) | EMG-RAG | 2024 | EMNLP | Indexing | [PDF](http://arxiv.org/pdf/2409.19401v1.pdf) | 
| [Personalized Graph-Based Retrieval for Large Language Models](http://arxiv.org/abs/2404.05970v1) | PGraphRAG | 2025 | arxiv | Indexing | [PDF](http://arxiv.org/pdf/2404.05970v1.pdf) | 
| [MeMemo: on-device retrieval augmentation for private and personalized text generation](http://arxiv.org/abs/2505.00263v1) | MeMemo | 2024 | SIGIR | Dense | [PDF](http://arxiv.org/pdf/2505.00263v1.pdf) | 
| [RECAP: retrieval-enhanced context-aware prefix encoder for personalized dialogue response generation](http://arxiv.org/abs/2408.02271v1) | RECAP | 2023 | arxiv | Dense | [PDF](http://arxiv.org/pdf/2408.02271v1.pdf) | 
| [Learning retrieval augmentation for personalized dialogue generation](http://arxiv.org/abs/2406.18847v1) | LAPDOG | 2024 | arxiv | Dense | [PDF](http://arxiv.org/pdf/2406.18847v1.pdf) | 
| [Partner matters! an empirical study on fusing personas for personalized response selection in retrieval-based chatbots](http://arxiv.org/abs/2310.06390v1) | | 2021 | SIGIR | Dense | [PDF](http://arxiv.org/pdf/2310.06390v1.pdf) | 
| [Personalm: Language model personalization via domain-distributed span aggregated k-nearest n-gram retrieval augmentation](http://arxiv.org/abs/2304.11406v4) | PersonaLM | 2023 | EMNLP | Dense | [PDF](http://arxiv.org/pdf/2304.11406v4.pdf) | 
| [A personalized dense retrieval framework for unified information access](http://arxiv.org/abs/2304.13654v1) | UIA | 2023 | SIGIR | Dense | [PDF](http://arxiv.org/pdf/2304.13654v1.pdf) | 
| [Personalized retrieval over millions of items](http://arxiv.org/abs/1812.04407v1) | XPERT | 2023 | SIGIR | Dense | [PDF](http://arxiv.org/pdf/1812.04407v1.pdf) | 
| [Towards personalized and semantic retrieval: An end-to-end solution for e-commerce search via embedding learning](http://arxiv.org/abs/2006.02282v3) | DPSR | 2020.0 | SIGIR | Dense | [PDF](http://arxiv.org/pdf/2006.02282v3.pdf) | 
| [Learning a fine-grained review-based transformer model for personalized product search](http://arxiv.org/abs/2005.08936v1) | RTM | 2021 | SIGIR | Dense | [PDF](http://arxiv.org/pdf/2005.08936v1.pdf) | 
| [Pearl: Personalizing large language model writing assistants with generation-calibrated retrievers](http://arxiv.org/abs/2306.16641v1) | Pearl | 2023 | arxiv | Dense | [PDF](http://arxiv.org/pdf/2306.16641v1.pdf) | 
| [Explainable recommendation with personalized review retrieval and aspect learning](http://arxiv.org/abs/2508.20312v2) | ERRA | 2023 | arxiv | Dense | [PDF](http://arxiv.org/pdf/2508.20312v2.pdf) | 
| [Efficient llm contextualization with user embeddings](http://arxiv.org/abs/2402.13598v2) | USER-LLM | 2024 | arxiv | Dense | [PDF](http://arxiv.org/pdf/2402.13598v2.pdf) | 
| [Integrating summarization and retrieval for enhanced personalization via large language models](http://arxiv.org/abs/2310.20081v1) | PAG | 2023 | arxiv | Sparse | [PDF](http://arxiv.org/pdf/2310.20081v1.pdf) | 
| [Personalized Graph-Based Retrieval for Large Language Models](http://arxiv.org/abs/2404.05970v1) | Au et al. | 2025 | arxiv | Sparse | [PDF](http://arxiv.org/pdf/2404.05970v1.pdf) | 
| [Unims-rag: A unified multi-source retrieval-augmented generation for personalized dialogue systems](http://arxiv.org/abs/2204.08128v1) | UniMS-RAG | 2024 | arxiv | Sparse | [PDF](http://arxiv.org/pdf/2204.08128v1.pdf) | 
| [Toward personalized answer generation in e-commerce via multi-perspective preference modeling](http://arxiv.org/abs/2112.13556v1) | Deng et al. | 2022 | TOIS | Sparse | [PDF](http://arxiv.org/pdf/2112.13556v1.pdf) | 
| [Doing personal laps: Llm-augmented dialogue construction for personalized multi-session conversational search](http://arxiv.org/abs/2405.03480v1) | LAPS | 2024 | SIGIR | Prompt-based | [PDF](http://arxiv.org/pdf/2405.03480v1.pdf) | 
| [Towards unified multi-modal personalization: Large vision-language models for generative recommendation and beyond](http://arxiv.org/abs/2309.13885v2) | UniMP | 2024 | arxiv | Prompt-based | [PDF](http://arxiv.org/pdf/2309.13885v2.pdf) | 
| [HEART-felt Narratives: Tracing Empathy and Narrative Style in Personal Stories with LLMs](http://arxiv.org/abs/2405.17633v2) | Shen et al. | 2024 | arxiv | Prompt-based | [PDF](http://arxiv.org/pdf/2405.17633v2.pdf) | 
| [Optimization methods for personalizing large language models through retrieval augmentation](http://arxiv.org/abs/2404.05970v1) | Salemi et al. | 2024 | SIGIR | Others | [PDF](http://arxiv.org/pdf/2404.05970v1.pdf) | 
| [PersonalTM: Transformer memory for personalized retrieval](http://arxiv.org/abs/2402.16288v1) | PersonalTM | 2023 | SIGIR | Others | [PDF](http://arxiv.org/pdf/2402.16288v1.pdf) | 
| [Personalized LoRA for human-centered text understanding](http://arxiv.org/abs/2403.06208v1) | Zhang et al. | 2024 | AAAI | Others | [PDF](http://arxiv.org/pdf/2403.06208v1.pdf) | 
| [PersonaRAG: Enhancing Retrieval-Augmented Generation Systems with User-Centric Agents](http://arxiv.org/abs/2508.03680v1) | PersonaRAG | 2024 | arxiv | Post-retrieval | [PDF](http://arxiv.org/pdf/2508.03680v1.pdf) | 
| [Unims-rag: A unified multi-source retrieval-augmented generation for personalized dialogue systems](http://arxiv.org/abs/2204.08128v1) | UniMS-RAG | 2024 | arxiv | Post-retrieval | [PDF](http://arxiv.org/pdf/2204.08128v1.pdf) | 
| [Learning to Rank for Multiple Retrieval-Augmented Models through Iterative Utility Maximization](http://arxiv.org/abs/2410.09942v2) | Salemi and Zamani | 2024 | arxiv | Post-retrieval | [PDF](http://arxiv.org/pdf/2410.09942v2.pdf) | 
| [Rehearse With User: Personalized Opinion Summarization via Role-Playing based on Large Language Models](http://arxiv.org/abs/2503.00449v1) | Zhang et al. | 2025 | arxiv | Post-retrieval | [PDF](http://arxiv.org/pdf/2503.00449v1.pdf) | 
| [Fit-rag: black-box rag with factual information and token reduction](http://arxiv.org/abs/2403.14374v1) | FIT-RAG | 2024 | arxiv | Post-retrieval | [PDF](http://arxiv.org/pdf/2403.14374v1.pdf) | 

#### 🤖 Agent

| Paper | Abbr. | Year | Source | Tag | PDF | 
|----------|------|------|------|------|--------|
| [Conversational health agents: A personalized llm-powered agent framework](http://arxiv.org/abs/2310.02374v5) | | 2023 | arxiv | Agent | [PDF](http://arxiv.org/pdf/2310.02374v5.pdf) | 
| [Rolellm: Benchmarking, eliciting, and enhancing role-playing abilities of large language models](http://arxiv.org/abs/2410.09541v1) | RoleLLM | 2023 | arxiv | Agent | [PDF](http://arxiv.org/pdf/2410.09541v1.pdf) | 
| [Character-llm: A trainable agent for role-playing](http://arxiv.org/abs/2509.12484v1) | Character-LLM | 2023 | arxiv | Agent | [PDF](http://arxiv.org/pdf/2509.12484v1.pdf) | 
| [Incharacter: Evaluating personality fidelity in role-playing agents through psychological interviews](http://arxiv.org/abs/2310.17976v4) | | 2023 | arxiv | Agent | [PDF](http://arxiv.org/pdf/2310.17976v4.pdf) | 
| [Mmrole: A comprehensive framework for developing and evaluating multimodal role-playing agents](http://arxiv.org/abs/2408.04203v2) | Mmrole | 2024 | arxiv | Agent | [PDF](http://arxiv.org/pdf/2408.04203v2.pdf) | 
| [Capturing minds, not just words: Enhancing role-playing language models with personality-indicative data](http://arxiv.org/abs/2406.18921v3) | | 2024 | arxiv | Agent | [PDF](http://arxiv.org/pdf/2406.18921v3.pdf) | 
| [Enabling conversational interaction with mobile ui using large language models](http://arxiv.org/abs/2209.08655v2) | | 2023 | CHI | Agent | [PDF](http://arxiv.org/pdf/2209.08655v2.pdf) | 
| [Neeko: Leveraging dynamic lora for efficient multi-character role-playing agent](http://arxiv.org/abs/2509.17676v1) | Neeko | 2024 | arxiv | Agent | [PDF](http://arxiv.org/pdf/2509.17676v1.pdf) | 
| [Voyager: An open-ended embodied agent with large language models](http://arxiv.org/abs/2406.16294v1) | VOYAGER | 2023 | arxiv | Agent | [PDF](http://arxiv.org/pdf/2406.16294v1.pdf) | 
| [Large language models empowered personalized web agents](http://arxiv.org/abs/2410.17236v2) | PUMA | 2025 | WWW | Agent | [PDF](http://arxiv.org/pdf/2410.17236v2.pdf) | 
| [Language models as zero-shot planners: Extracting actionable knowledge for embodied agents](http://arxiv.org/abs/2406.16294v1) | | 2022 | PMLR | Agent | [PDF](http://arxiv.org/pdf/2406.16294v1.pdf) | 
| [Generative agents: Interactive simulacra of human behavior](http://arxiv.org/abs/2208.04024v1) | | 2023 | UIST | Agent | [PDF](http://arxiv.org/pdf/2208.04024v1.pdf) | 
| [Agents meet okr: An object and key results driven agent system with hierarchical self-collaboration and self-evaluation](http://arxiv.org/abs/2311.16542v1) | OKR-Agent | 2023 | arxiv | Agent | [PDF](http://arxiv.org/pdf/2311.16542v1.pdf) | 
| [Investigating the Personality Consistency in Quantized Role-Playing Dialogue Agents](http://arxiv.org/abs/2502.03821v1) | | 2024 | EMNLP | Agent | [PDF](http://arxiv.org/pdf/2502.03821v1.pdf) | 
| [Socialbench: Sociality evaluation of role-playing conversational agents](http://arxiv.org/abs/2503.17460v2) | Socialbench | 2024 | arxiv | Agent | [PDF](http://arxiv.org/pdf/2503.17460v2.pdf) | 
| [PersonaAgent: When Large Language Model Agents Meet Personalization at Test Time](http://arxiv.org/abs/2506.06254v1) | PersonaAgent | 2025 | arxiv | Agent | [PDF](http://arxiv.org/pdf/2506.06254v1.pdf) | 
| [Personax: A recommendation agent oriented user modeling framework for long behavior sequence](http://arxiv.org/abs/2503.02398v2) | Personax | 2025 | arxiv | Agent | [PDF](http://arxiv.org/pdf/2503.02398v2.pdf) | 
| [Llm-powered multi-agent framework for goal-oriented learning in intelligent tutoring system](http://arxiv.org/abs/2502.16613v1) | GenMentor | 2025 | arxiv | Agent | [PDF](http://arxiv.org/pdf/2502.16613v1.pdf) | 
| [Multi-agent collaboration mechanisms based on distributed online meta-learning for mass personalization](http://arxiv.org/abs/2411.07094v2) | | 2025 | JIII | Agent | [PDF](http://arxiv.org/pdf/2411.07094v2.pdf) | 
| [Agent4Ranking: Semantic Robust Ranking via Personalized Query Rewriting Using Multi-Agent LLMs](http://arxiv.org/abs/2411.14739v1) | Agent4Ranking | 2025 | TOIS | Agent | [PDF](http://arxiv.org/pdf/2411.14739v1.pdf) | 
| [One Size doesn't Fit All: A Personalized Conversational Tutoring Agent for Mathematics Instruction](http://arxiv.org/abs/2502.12633v2) | PACE | 2025 | WWW | Agent | [PDF](http://arxiv.org/pdf/2502.12633v2.pdf) | 
| [MAP: Multi-user Personalization with Collaborative LLM-powered Agents](http://arxiv.org/abs/2506.11803v2) | MAP | 2025 | CHI | Agent | [PDF](http://arxiv.org/pdf/2506.11803v2.pdf) | 
| [Evaluating personalized tool-augmented llms from the perspectives of personalization and proactivity](http://arxiv.org/abs/2503.00771v2) | | 2025 | arxiv | Agent | [PDF](http://arxiv.org/pdf/2503.00771v2.pdf) | 

#### 🧠 Memory

| Paper | Abbr. | Year | Source | Tag | PDF | 
|----------|------|------|------|------|--------|
| [Align on the fly: Adapting chatbot behavior to established norms](http://arxiv.org/abs/2312.15907v1) | OPO | 2023 | arxiv | memory | [PDF](http://arxiv.org/pdf/2312.15907v1.pdf) | 
| [LLM-based medical assistant personalization with short-and long-term memory coordination](http://arxiv.org/abs/2309.11696v3) | MALP | 2023 | arxiv | memory | [PDF](http://arxiv.org/pdf/2309.11696v3.pdf) | 
| [Memory-assisted prompt editing to improve GPT-3 after deployment](http://arxiv.org/abs/2201.06009v7) | MemPrompt | 2022 | arxiv | memory | [PDF](http://arxiv.org/pdf/2201.06009v7.pdf) | 


## ⚙️ Training-time Personalized Alignment
#### 🔧 Fine-tune

| Paper | Abbr. | Year | Source | Tag | PDF | 
|----------|------|------|------|------|--------|
| [Fdlora: Personalized federated learning of large language model via dual lora tuning](http://arxiv.org/abs/2406.07925v1) | Fdlora | 2024 | arxiv | PEFT | [PDF](http://arxiv.org/pdf/2406.07925v1.pdf) | 
| [Democratizing large language models via personalized parameter-efficient fine-tuning](http://arxiv.org/abs/2402.04401v3) | | 2024 | arxiv | PEFT | [PDF](http://arxiv.org/pdf/2402.04401v3.pdf) | 
| [Persoma: Personalized soft prompt adapter architecture for personalized language prompting](http://arxiv.org/abs/2408.00960v1) | Persoma | 2024 | arxiv | PEFT | [PDF](http://arxiv.org/pdf/2408.00960v1.pdf) | 
| [Embedding-to-Prefix: Parameter-Efficient Personalization for Pre-Trained Large Language Models](http://arxiv.org/abs/2309.14726v2) | | 2025 | arxiv | PEFT | [PDF](http://arxiv.org/pdf/2309.14726v2.pdf) | 
| [FaST: Feature-aware Sampling and Tuning for Personalized Preference Alignment with Limited Data](http://arxiv.org/abs/2508.04698v1) | FaST | 2025 | arxiv | PEFT | [PDF](http://arxiv.org/pdf/2508.04698v1.pdf) | 
| [Personalized large language models through parameter efficient fine-tuning techniques](http://arxiv.org/abs/2506.05316v1) | | 2024 | SIGIR | PEFT | [PDF](http://arxiv.org/pdf/2506.05316v1.pdf) | 
| [Personalized pieces: Efficient personalized large language models through collaborative efforts](http://arxiv.org/abs/2406.10471v3) | PER-PCS | 2024 | arxiv | PEFT | [PDF](http://arxiv.org/pdf/2406.10471v3.pdf) | 
| [DiffuseKronA: A Parameter Efficient Fine-tuning Method for Personalized Diffusion Models](http://arxiv.org/abs/2502.05895v1) | DiffuseKronA | 2024 | WACV | PEFT | [PDF](http://arxiv.org/pdf/2502.05895v1.pdf) | 
| [Improving Personalized Sentiment Representation with Knowledge-enhanced and Parameter-efficient Layer Normalization](http://arxiv.org/abs/1810.06645v1) | E2LN | 2024 | COLING | PEFT | [PDF](http://arxiv.org/pdf/1810.06645v1.pdf) | 
| [Customizing large language model generation style using parameter-efficient finetuning](http://arxiv.org/abs/2409.04574v1) | | 2024 | arxiv | PEFT | [PDF](http://arxiv.org/pdf/2409.04574v1.pdf) | 
| [Personalize Your LLM: Fake it then Align it](http://arxiv.org/abs/2503.01048v3) | CHAMELEON | 2025 | arxiv | PEFT | [PDF](http://arxiv.org/pdf/2503.01048v3.pdf) | 
| [FedMCP: parameter-efficient federated learning with model-contrastive personalization](http://arxiv.org/abs/2206.13190v1) | FedMCP | 2024 | arxiv | PEFT | [PDF](http://arxiv.org/pdf/2206.13190v1.pdf) | 
| [Efficient model-agnostic alignment via bayesian persuasion](http://arxiv.org/abs/2405.18718v1) | | 2024 | arxiv | PEFT | [PDF](http://arxiv.org/pdf/2405.18718v1.pdf) | 
| [Do llms understand user preferences? evaluating llms on user rating prediction](http://arxiv.org/abs/2305.06474v1) | | 2023 | arxiv | FULL | [PDF](http://arxiv.org/pdf/2305.06474v1.pdf) | 
| [Teach LLMs to Personalize--An Approach inspired by Writing Education](http://arxiv.org/abs/2308.07968v1) | | 2023 | arxiv | FULL | [PDF](http://arxiv.org/pdf/2308.07968v1.pdf) | 

#### 📌 Embedding Learning

| Paper | Abbr. | Year | Source | Tag | PDF | 
|----------|------|------|------|------|--------|
| [Personalized steering of large language models: Versatile steering vectors through bi-directional preference optimization](http://arxiv.org/abs/2406.00045v2) | | 2024 | NIPS | | [PDF](http://arxiv.org/pdf/2406.00045v2.pdf) | 
| [User-llm: Efficient llm contextualization with user embeddings](http://arxiv.org/abs/2402.13598v2) | UserLLM | 2024 | arxiv | contextual embedding | [PDF](http://arxiv.org/pdf/2402.13598v2.pdf) | 
| [Distributional preference learning: Understanding and accounting for hidden context in RLHF](http://arxiv.org/abs/2312.08358v2) | | 2024 | ICLR | Latent Variable Encoding | [PDF](http://arxiv.org/pdf/2312.08358v2.pdf) | 
| [Enhancing social media personalization: dynamic user profile embeddings and multimodal contextual analysis using transformer models](http://arxiv.org/abs/2407.07925v1) | | 2024 | arxiv | contextual embedding | [PDF](http://arxiv.org/pdf/2407.07925v1.pdf) | 
| [Personalized query expansion with contextual word embeddings](http://arxiv.org/abs/2103.05256v1) | | 2023 | TOIS | contextual embedding | [PDF](http://arxiv.org/pdf/2103.05256v1.pdf) | 
| [User embedding model for personalized language prompting](http://arxiv.org/abs/2408.00960v1) | UEM | 2024 | arxiv | contextual embedding | [PDF](http://arxiv.org/pdf/2408.00960v1.pdf) | 
| [Knowledge-augmented large language models for personalized contextual query suggestion](http://arxiv.org/abs/2311.06318v2) | | 2024 | WWW | contextual embedding | [PDF](http://arxiv.org/pdf/2311.06318v2.pdf) | 
| [DLVGen: A Dual Latent Variable Approach to Personalized Dialogue Generation](http://arxiv.org/abs/2111.11363v1) | | 2022 | ICAART | Latent Variable Encoding | [PDF](http://arxiv.org/pdf/2111.11363v1.pdf) | 
| [Miracle: Towards Personalized Dialogue Generation with Latent-Space Multiple Personal Attribute Control](http://arxiv.org/abs/2310.18342v1) | | 2023 | EMNLP | Latent Variable Encoding | [PDF](http://arxiv.org/pdf/2310.18342v1.pdf) | 
| [PIE: A Personalized Information Embedded model for text-based depression detection](http://arxiv.org/abs/2408.03648v1) | PIE | 2024 | IPM | contextual embedding | [PDF](http://arxiv.org/pdf/2408.03648v1.pdf) | 
| [Morpheus: Modeling role from personalized dialogue history by exploring and utilizing latent space](http://arxiv.org/abs/2407.02345v1) | Morpheus | 2024 | arxiv | Latent Variable Encoding | [PDF](http://arxiv.org/pdf/2407.02345v1.pdf) | 

#### 👥 Learning From Human Feedback

| Paper | Abbr. | Year | Source | Tag | PDF | 
|----------|------|------|------|------|--------|
| [RLHF fine-tuning of LLMs for alignment with implicit user feedback in conversational recommenders](http://arxiv.org/abs/2508.05289v1) | | 2025 | arxiv | RLHF | [PDF](http://arxiv.org/pdf/2508.05289v1.pdf) | 
| [Personalized language modeling from personalized human feedback](http://arxiv.org/abs/2508.10695v1) | P-RLHF | 2024 | arxiv | RLHF | [PDF](http://arxiv.org/pdf/2508.10695v1.pdf) | 
| [MaxMin-RLHF: Alignment with diverse human preferences](http://arxiv.org/abs/2405.14705v1) | MaxMin-RLHF | 2024 | arxiv | RLHF | [PDF](http://arxiv.org/pdf/2405.14705v1.pdf) | 
| [Personalizing reinforcement learning from human feedback with variational preference learning](http://arxiv.org/abs/2402.05133v3) | | 2024 | NIPS | RLHF | [PDF](http://arxiv.org/pdf/2402.05133v3.pdf) | 
| [A Shared Low-Rank Adaptation Approach to Personalized RLHF](http://arxiv.org/abs/2503.19201v1) | | 2025 | AISTATS | RLHF | [PDF](http://arxiv.org/pdf/2503.19201v1.pdf) | 
| [Fedrlhf: A convergence-guaranteed federated framework for privacy-preserving and personalized rlhf](http://arxiv.org/abs/2412.15538v2) | FedRLHF | 2024 | arxiv | RLHF | [PDF](http://arxiv.org/pdf/2412.15538v2.pdf) | 
| [Rlhf from heterogeneous feedback via personalization and preference aggregation](http://arxiv.org/abs/2405.00254v2) | | 2024 | arxiv | RLHF | [PDF](http://arxiv.org/pdf/2405.00254v2.pdf) | 
| [Value augmented sampling for language model alignment and personalization](http://arxiv.org/abs/2405.06639v1) | | 2024 | arxiv | RLHF | [PDF](http://arxiv.org/pdf/2405.06639v1.pdf) | 
| [Personalized soups: Personalized large language model alignment via post-hoc parameter merging](http://arxiv.org/abs/2310.11564v1) | RLPHF | 2023 | arxiv | RLHF | [PDF](http://arxiv.org/pdf/2310.11564v1.pdf) | 
| [Fine-grained human feedback gives better rewards for language model training](http://arxiv.org/abs/2306.01693v2) | Fine-grained RLHF | 2023 | NIPS | RLHF | [PDF](http://arxiv.org/pdf/2306.01693v2.pdf) | 
| [Robust Multi-Objective Preference Alignment with Online DPO](http://arxiv.org/abs/2406.05534v1) | MO-ODPO | 2025 | AAAI | DPO | [PDF](http://arxiv.org/pdf/2406.05534v1.pdf) | 
| [Beyond one-preference-for-all: Multi-objective direct preference optimization](http://arxiv.org/abs/2508.07638v1) | MODPO | 2023 | arxiv | DPO | [PDF](http://arxiv.org/pdf/2508.07638v1.pdf) | 
| [Instantly learning preference alignment via in-context DPO](http://arxiv.org/abs/2505.01706v1) | | 2025 | NAACL | DPO | [PDF](http://arxiv.org/pdf/2505.01706v1.pdf) | 
| [Towards robust alignment of language models: Distributionally robustifying direct preference optimization](http://arxiv.org/abs/2407.07880v2) | Dr. DPO | 2024 | arxiv | DPO | [PDF](http://arxiv.org/pdf/2407.07880v2.pdf) | 
| [Multi-Preference Lambda-weighted Listwise DPO for Dynamic Preference Alignment](http://arxiv.org/abs/2510.01540v1) | | 2025 | arxiv | DPO | [PDF](http://arxiv.org/pdf/2510.01540v1.pdf) | 
| [Personalized language modeling from personalized human feedback](http://arxiv.org/abs/2508.10695v1) | P-DPO | 2024 | arxiv | DPO | [PDF](http://arxiv.org/pdf/2508.10695v1.pdf) | 
| [DreamBoothDPO: Improving Personalized Generation using Direct Preference Optimization](http://arxiv.org/abs/2507.01479v1) | DreamBoothDPO | 2025 | arxiv | DPO | [PDF](http://arxiv.org/pdf/2507.01479v1.pdf) | 
| [alpha-DPO: Adaptive Reward Margin is What Direct Preference Optimization Needs](http://arxiv.org/abs/2502.08922v1) | alpha-DPO | 2024 | arxiv | DPO | [PDF](http://arxiv.org/pdf/2502.08922v1.pdf) | 
| [β-DPO: Direct Preference Optimization with Dynamic β](http://arxiv.org/abs/1812.05748v4) | β-DPO | 2024 | NIPS | DPO | [PDF](http://arxiv.org/pdf/1812.05748v4.pdf) | 

#### 🔄 Reinforcement Learning

| Paper | Abbr. | Year | Source | Tag | PDF | 
|----------|------|------|------|------|--------|
| [Cultivating Helpful, Personalized, and Creative AI Tutors: A Framework for Pedagogical Alignment using Reinforcement Learning](http://arxiv.org/abs/2507.20335v1) | | 2025 | arxiv | GRPO | [PDF](http://arxiv.org/pdf/2507.20335v1.pdf) | 
| [GroupAligner: A Deep Reinforcement Learning with Domain Adaptation for Social Group Alignment](http://arxiv.org/abs/1812.07452v1) | | 2023 | WWW | deep-reinforment learning | [PDF](http://arxiv.org/pdf/1812.07452v1.pdf) | 
| [Optimizing Safe and Aligned Language Generation: A Multi-Objective GRPO Approach](http://arxiv.org/abs/2503.21819v1) | | 2025 | arxiv | GRPO | [PDF](http://arxiv.org/pdf/2503.21819v1.pdf) | 
| [Fine-Tuning a Large Language Model with Reinforcement Learning for Educational Question Generation](http://arxiv.org/abs/2212.03869v1) | | 2024 | AIED | on-policy RL | [PDF](http://arxiv.org/pdf/2212.03869v1.pdf) | 
| [From Problem-Solving to Teaching Problem-Solving: Aligning LLMs with Pedagogy using Reinforcement Learning](http://arxiv.org/abs/2505.15607v2) | | 2025 | arxiv | on-policy RL | [PDF](http://arxiv.org/pdf/2505.15607v2.pdf) | 

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Tomsawyerhu/Awesome-Personalized-Alignment&type=Date)](https://star-history.com/#Tomsawyerhu/Awesome-Personalized-Alignment&Date)


