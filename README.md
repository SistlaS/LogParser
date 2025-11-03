# LogParser

In this project, we tackled the problem of schema and key-value extraction from structured system
logs using large language models. We constructed a high-quality annotated dataset by replacing
8 generic templates with semantically meaningful field names based on each log’s EventId. This
dataset served both as ground truth for evaluation and as an example repository for prompting.
We evaluated multiple prompting strategies using GPT-4o and found that:
• Zero-shot prompting performed poorly due to the domain-specific nature of the task and lack
of structural context.
• Few-shot prompting using an exact example from the same EventId yielded near-perfect
results.
• Few-shot prompting using a semantically similar log found via FAISS provided a realistic and
effective fallback when exact matches were unavailable.
We conclude that LLMs, when guided with structured examples, are powerful tools for log
parsing tasks. Our results highlight the value of hybrid systems that combine classical techniques
(template databases, embedding similarity) with modern language models for real-world system
understanding tasks.

Project contains:
- **Zero-shot** and **few-shot** learning for parsing log lines
- **Template normalization**
- **Embedding generation** for clustering or similarity search

---
##Features
Extract log templates and variable values  
Normalize templates for clustering  
Generate embeddings with `text-embedding-3-small`  
Few-shot learning for better generalization  
Modular and easily extensible code
---

