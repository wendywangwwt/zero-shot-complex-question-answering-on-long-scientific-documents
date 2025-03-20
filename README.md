# Zero-Shot Complex Question-Answering on Long Scientific Documents
This is the implementation of paper [Zero-Shot Complex Question-Answering on Long Scientific Documents](https://arxiv.org/html/2503.02695v1).

## MLPsych Dataset
The MLPsych dataset consists of 151 extractive QA pairs extracted from 52 social psychological papers published on English peer-reviewed journals. Each paper is annotated with the answer to at least 2 of the following 4 questions:

|No.|Question|Answer Type|Challenges|Complexity|# Spans|# Words|
|-|-|-|-|-|-|-|
|1|What machine learning or natural language processing techniques were used?|Entity|Multi-Span|Low|2.78|7.56|
|2|What software was used to perform machine learning or natural language processing techniques?|Entity|Multi-Span, Unanswerable|Medium|1.83|2.60|
|3|What was the research question that machine learning or natural language processing techniques were used to answer?|Phrase|Multi-Span, Long Answer|High|1.21|26.00|
|4|What were machine learning or natural language processing techniques used for?|Phrase|Multi-Span, Multi-Hop|High|2.78|27.89|

## Question-Answering Model Pipelines

### Install
Use the `requirements.txt`:
```
pip install -r requirements.txt
```

### Run Inference
Example command:
```
python model/main.py -pt MLPsych/df_text.pk -ct text_cleaned -pq questions.json --dir_output pred/MLPsych --device 0
```

*Currently the code requires one GPU device to run.*

### Questions Config
We load questions from a json dictionary, with key as question id/name and value as a list (`[<question text>, [<model>], [<special setup if needed>]]`).

The inference setup for the best results used in our experiments is as follows:
```
{"1": ["What machine learning or natural language processing techniques were used?",
       ["deepset/deberta-v3-large-squad2",
        "ahotrod/albert_xxlargev1_squad2_512"],
       ["rag","Extract the detailed machine learning or natural language processing techniques mentioned in the following sentence, such as a specific algorithm name, and return a python list of techniques. Avoid using high-level categories like machine learning or natural language processing, unless there is no specific algorithm name mentioned in the sentence. Standardize the categories. Use double quotes for each category. Directly return the python-readable flat list of techniques. Do not explain or add anything else. Sentence: ","mergeTrue"]],
 "2": ["What software was used to perform machine learning or natural language processing techniques?",
       ["ahotrod/electra_large_discriminator_squad2_512",
        "deepset/roberta-large-squad2"],
       ["rag","Extract only the software or tool mentioned in the following text snippet, if there is any, such as programming language, package, platform, or chatbot, and return a python list of software or tool. Skip machine learning or natural language processing techniques. Do not explain or add anything else. Only return the python list. Text: ","mergeFalse"]],
 "3": ["What was the research question that machine learning or natural language processing techniques were used to answer?",
       ["deepset/deberta-v3-large-squad2",
        "ahotrod/albert_xxlargev1_squad2_512"],
       []],
 "4": ["What was __PLACEHOLDER__ used for?",
       ["deepset/deberta-v3-large-squad2"],
       ["msh","1"]]}
```

In question-specific settings:
- `["rag",<prompt>,<arg>]`: RAG-enhanced multi-span entity extraction; argument `mergeFalse` indicates not to merge predicted answers using implemented strategies (see paper), which is important for questions where key answers should not be mixed up (e.g., for ML software, R should not be merged with other software that has letter `R` in the name)
- `["msh",<question id>]`: Multi-span multi-hop as multi-single-hop, and the question id refers to the answer of the question which will be used as the bridge entity to create sub-questions for the current one; note that in the text **there needs to be a `__PLACEHOLDER__` token** to be replaced by the extracted bridge entities
