# Personalized ChatBot

HSE University 2024-2026

We present the approach that is aimed at effectively expanding the context through integrating a database of associative memory into the pipeline. In order to improve long-term memory and personalization we have utilized methods close to Retrieval-Augmented Generation (RAG). Our method uses a multi-agent pipeline with a cold-start agent for initial interactions, a fact extraction agent to process user inputs, an associative memory agent for storing and retrieving context, and a generation agent for replying to user’s queries.Evaluation results show promising results: a 41% accuracy improvement over the base Gemma3 model (from 16% to 57%). Hence, with our approach, we demonstrate that personalized chatbots can bypass LLM memory limitations while increasing information reliability under the conditions of limited context and memory.

## Installation and usage

```bash
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
(venv) $ echo YOUR_TG_TOKEN > .token
(venv) $ python3 telegram_bot.py  # this runs application and makes bot avaliable in Telegram
```

## Usage inside your code

```python
from src import ChatBot

chatbot = ChatBot(use_vllm=False) # True if you want to use vLLM for 3x inference speed boost

print(chatbot.response("""
I am 32. I do not want a job. I play video games all day. I still live at home with my parents. My favorite drink is iced coffee.
I have a black belt in karate. I m in a jazz band and play the saxophone. I vacation along lake michigan every summer.
"""))

print(chatbot.response("How old am I?"))
```

## Changelog

- **feb 2026**: finalized Telegram bot

- **jan 2026**: developed a Qdrant database, moved the generator on vllm, continued to work on Telegram bot.

- **oct-dec 2025**: ran experiments with FAISS and experiments for improvement fact extraction module's quality, started to develop an application for ChatBot in Telegram (python telegram bot api); tried to move our ChatBot to Triton Inference Server (failed, abandoned), langchain (failed, abandoned).

- **may-sep 2025**: abandoned the idea of fine-tuning the models for a generation agent, chose prompts instead, worked on the research paper and submitted it to RANLP 2025 Student Research Workshop, decided to work on improvement of the ChatBot in general.

- **april 2025**: started working on the research paper;
    - *cold start* - continued to develop approaches for matching similar personas (encoder training), evaluated of the approaches, selected the most effective and fastest one, implementated algorithms for selecting similar personas (baseline);
    - *fact extraction* - ran experiments with improving the quality of baselines, trained the seq2seq model on the task of fact extraction, generated training data, chose evaluation metrics (BLEU, WER, LER, NLI);
    - *associative memory* - continued to run experiments and work on collision resolution: finalized heuristic algorithm, ran it with different encoders, implemented the "ChatBot associative memory" module in the pipeline;
    -  *generation*  - continued to work on prompts, experimented with fine-tuning gemma-3-1b-it, qwen-2.5-1.5b-it and qwen-2.5.-0.5b-it (SFT, LoRA), implemented the "generation" module in the pipeline.

- **march 2025**: 
    - *cold start* - was developing approaches for matching similar personas: simple algorithms, GNN;
    - *fact extraction* - ran experiments with different methods of fact extraction, created the first versions of targets in the JSON triplets fortmat for training seq2seq model;
    - *associative memory* - continued to test the baselines (BM25, sentence transformers), worked on collision resolution (collecting dataset for quality measurement, running experiments with own heuristic algorithm);
    -  *generation*  - tried to work with Gemma-3-1b-it, wrote and tested different prompts (single-turn and multi-turn approaches), worked with the Multi-Session Chat dataset (preprocessing for future experiments).

- **feb 2025**:
    - *cold start* - ran experiments with the "Synthetic Persona Chat" dataset, encoded personas with different embedding models, created a heterogeneous and homogeneous graph of persona's relations based on embeddings;
    - *fact extraction* - preprocessed datasets, developed the agent's pipeline,tested the baselines: regular rules, syntactic tree (Spacy), extractive summarization (TextRank, Bert), abstractive summarization (T5, Bart);
    - *associative memory* - tested the baselines: tf-idf, count vectorizers (for raking), prompt engineering (collision resolution)
    -  *generation* - ran experiments with llama-3.2-1b-it, tested different prompting techniques.

- **dec 2024 - jan 2025**: assigned roles in project, reviewed research papers on the topic

## License

Our work is distributed under the MIT license.

## Citation

```txt
[Personalizing chatbot communication with associative memory](https://aclanthology.org/2025.ranlp-stud.8/)
Kirill Soloshenko, Alexandra Shatalina, Marina Sevostyanova, Elizaveta Kornilova, and Konstantin Zaitsev. 2025. Personalizing chatbot communication with associative memory. In Proceedings of the 9th Student Research Workshop associated with the International Conference Recent Advances in Natural Language Processing, pages 62–69, Varna, Bulgaria. INCOMA Ltd., Shoumen, Bulgaria.
```

```bibtex
@inproceedings{soloshenko-etal-2025-personalizing,
    title = "Personalizing chatbot communication with associative memory",
    author = "Soloshenko, Kirill  and
      Shatalina, Alexandra  and
      Sevostyanova, Marina  and
      Kornilova, Elizaveta  and
      Zaitsev, Konstantin",
    editor = "Velichkov, Boris  and
      Nikolova-Koleva, Ivelina  and
      Slavcheva, Milena",
    booktitle = "Proceedings of the 9th Student Research Workshop associated with the International Conference Recent Advances in Natural Language Processing",
    month = sep,
    year = "2025",
    address = "Varna, Bulgaria",
    publisher = "INCOMA Ltd., Shoumen, Bulgaria",
    url = "https://aclanthology.org/2025.ranlp-stud.8/",
    pages = "62--69",
    abstract = "In our research paper we present the approach that is aimed at effectively expanding the context through integrating a database of associative memory into the pipeline. In order to improve long-term memory and personalization we have utilized methods close to Retrieval-Augmented Generation (RAG). Our method uses a multi-agent pipeline with a cold-start agent for initial interactions, a fact extraction agent to process user inputs, an associative memory agent for storing and retrieving context, and a generation agent for replying to user{'}s queries.Evaluation results show promising results: a 41{\%} accuracy improvement over the base Gemma3 model (from 16{\%} to 57{\%}). Hence, with our approach, we demonstrate that personalized chatbots can bypass LLM memory limitations while increasing information reliability under the conditions of limited context and memory."
}
```

## Team

- Alexandra Shatalina: a cold-start agent
- Elizaveta Kornilova: a fact extraction agent
- Kirill Soloshenko: an associative memory agent
- Marina Sevostyanova: a generation agent
