# Prompt Engineering for Linguistic Tasks

This work is inspired by the paper [*Applying Linguistic Expertise to LLMs for Educational Material Development in Indigenous Languages* by Vasselli et al. (2024)](https://aclanthology.org/2024.americasnlp-1.24.pdf). 

Link to paper's github: [JAJ-Americas2024](https://github.com/JVasselli/JAJ-Americas2024)

This project is designed to replicate prompt-based engineering to generate predictions for linguistic transformations. The implementation reads training and development data, finds relevant examples, creates prompts, and sends them to an OpenAI model for predictions.

## Pipeline Overview

The process begins by loading the data and ends with saving the generated predictions. Here's a step-by-step breakdown:

1. **Load Data**: Training and development datasets (in TSV format) are loaded using `pandas`. The training data serves as the source for extracting relevant examples for few-shot learning, and the development data is used to create queries for predictions.

2. **Find Relevant Examples**: For each row in the development dataset, the script selects the most relevant examples from the training data based on cosine similarity. This can be done using:
    - **TF-IDF Similarity**: Calculates traditional TF-IDF-based cosine similarity to determine example relevance.
    - **Sentence Transformer Similarity**: Uses a pre-trained Sentence Transformer model to compute embeddings and measure cosine similarity between training and development data.

3. **Create Prompt**: A prompt is generated by combining the relevant examples from the training data with the current source sentence and the desired transformation. The format typically includes linguistic examples, the current sentence, and the necessary change.

4. **Get Prediction**: The generated prompt is sent to OpenAI’s API (e.g., `gpt-4o-mini`) to obtain the prediction for the specified transformation.

5. **Save Predictions**: The generated predictions are appended to the development dataset and saved as a new TSV file.

## Files Included

The first two python scripts mentioned below as well as the `data/` files are supplied by and based upon [AmericasNLP 2024 shared tasks](https://github.com/AmericasNLP/americasnlp2024/tree/master/ST2_EducationalMaterials).

### 1. `baseline_edit_trees.py`
This script is responsible for creating edit trees from a training dataset and using them to predict changes for a given development dataset.

**Parameters**:
- `train.tsv`: The path to the training dataset in TSV format, containing `Source`, `Target`, and `Change` columns.
- `dev.tsv`: The path to the development dataset for which predictions are to be made.
- `dev-prediction.tsv`: The path to save the output predictions in TSV format.

**Workflow**:
- The script reads a training dataset and builds a set of edit trees, linking `Change` types to the edit trees that occur for those changes.
- It then processes the development dataset, making predictions by applying the most common edit tree for a given `Change` to the `Source` sentence.

### 2. `evaluate.py`
This script evaluates the performance of the predictions generated by `baseline_Edit_trees.py` or `llm.py`. It computes accuracy, BLEU, and ChrF scores.

**Parameters**:
- `dev-prediction.tsv`: The path to the file with predictions to evaluate, containing `Source`, `Target`, and `Predicted Target` columns.

**Metrics**:
- **Accuracy**: Measures the percentage of exact matches between `Target` and `Predicted Target`.
- **BLEU**: Measures the quality of the predictions based on n-gram overlap.
- **ChrF**: Measures similarity between `Target` and `Predicted Target` at the character level.

### 3. `llm.py`
This script uses a language model to generate predictions based on a provided training dataset and development dataset. The script employs cosine similarity to find the most relevant examples from the training data and uses those examples to create a prompt for the language model.

**Dependencies**:
- OpenAI API (`openai`)
- Sentence Transformers (`sentence_transformers`)
- scikit-learn (`sklearn`)
- dotenv (`dotenv`)

**Parameters**:
- `train.tsv`: Path to the training data file containing `Source`, `Target`, and `Change` columns.
- `dev.tsv`: Path to the development data file for which predictions are required.
- `results.tsv`: Path to save the output with predictions in TSV format.

**Environment Variables**:
- Requires an API key (`OPENAI_API_KEY`) to be set in an environment file (`.env`).

**Configuration**

- `TOP_N`: Number of relevant examples to retrieve for few-shot learning.
- `GPT_MODEL`: The OpenAI model to use for generating predictions (e.g., gpt-4o-mini).
- `USE_SENTENCE_TRANSFORMER`: Boolean flag to toggle between using TF-IDF or a Sentence Transformer model.
- `SENTENCE_TRANSFORMER_MODEL`: The Sentence Transformer model to use (e.g., 'all-MiniLM-L6-v2').

You can modify these constants directly in the script to suit your requirements.

## Prerequisites
- Python 3.7+
- Required Python packages (can be installed using `pip`):
  - `spacy`
  - `sentence_transformers`
  - `scikit-learn`
  - `pandas`
  - `dotenv`
  - `openai`
  - `sacrebleu`

## Setup
1. Clone the repository.
2. Install the required Python packages using the following command:
   ```
   pip install -r requirements.txt
   ```
3. Set up the `.env` file with your OpenAI API key.
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage Example
1. First, run `baseline_edit_trees.py` to generate baseline:
   ```bash
   python baseline_edit_trees.py guarani-train.tsv guarani-dev.tsv baseline.tsv
   ```
2. Evaluate the generated predictions using `evaluate.py`:
   ```bash
   python evaluate.py baseline.tsv
   ```
3. Use `llm.py` to generate predictions using a language model:
   ```bash
   python llm.py guarani-train.tsv guarani-dev.tsv results.tsv
   ```
4. Evaluate the predictions generated by `llm.py`:
   ```bash
   python evaluate.py results.tsv
   ```

## Notes
- The `llm.py` script allows flexibility in choosing similarity computation methods (TF-IDF or Sentence Transformer).

## Tag Breakdown:
For documentation purposes, here is a legend of the possible tags used in the `train.tsv` and `dev.tsv` files:
| Tag             | Meaning                                                       | Example                                             |
|-----------------|---------------------------------------------------------------|-----------------------------------------------------|
| TYPE:AFF        | Affirmative Phrase Type (a positive statement)                 | She is going to the market.                         |
| TYPE:NEG        | Negative Phrase Type (a negative statement)                    | She is not going to the market.                     |
| TYPE:IMP        | Imperative Phrase Type (a command or request)                  | Go to the market.                                   |
| SUBTYPE:DEC     | Declarative Subtype (a statement providing information)        | It is raining.                                      |
| SUBTYPE:INT     | Interrogative Subtype (a question)                             | Is it raining?                                      |
| MODE:ADVERS     | Adversative Mode (a statement that expresses contrast or conflict) | Although it is raining, we will still go out.       |
| MODE:EXH        | Exhortative Mode (an encouraging suggestion or urging someone) | Let’s all work together!                            |
| MODE:DES        | Optative Mode (expressing a wish or desire)                    | May you have a great day.                           |
| MODE:IND        | Indicative Mode (a statement of fact)                          | She walks every day.                                |
| MODE:POT        | Potential Mode (expressing possibility)                        | She could go to the market if it stops raining.     |
| MODE:SUB        | Subjunctive Mode (expressing a hypothetical situation or wish) | If I were rich, I would travel the world.           |
| TENSE:PRE_SIM   | Present Simple Tense (describes a current action or state)     | She walks to school.                                |
| TENSE:PAS_SIM   | Past Simple Tense (describes a completed action in the past)   | She walked to school.                               |
| TENSE:FUT_SIM   | Future Simple Tense (describes an action that will happen)     | She will walk to school.                            |
| TENSE:PRF_REM   | Perfect Remote Tense (describes an action that was completed long ago) | She had already eaten before we arrived.        |
| ASPECT:IPFV     | Imperfect Aspect (ongoing or habitual actions in the past)     | She was walking to school.                          |
| ASPECT:PFV      | Perfect Aspect (completed action)                              | She has finished her homework.                      |
| ASPECT:PRG      | Progressive Aspect (continuous action)                         | She is walking to school.                           |
| ASPECT:HAB      | Habitual Aspect (actions happening regularly)                  | She walks to school every day.                      |
| VOICE:ACT       | Active Voice (subject performs the action)                     | The cat chased the mouse.                           |
| VOICE:PAS       | Passive Voice (action is performed on the subject)             | The mouse was chased by the cat.                    |
| VOICE:MID       | Middle Voice (subject is both agent and recipient of the action) | She washed herself.                                 |
| PERSON:1_SI     | First Person Singular (referring to oneself)                   | I am reading.                                       |
| PERSON:1_PL     | First Person Plural (referring to a group including oneself)   | We are reading.                                     |
| PERSON:2_SI     | Second Person Singular (referring to the person spoken to, singular) | You are reading.                                 |
| PERSON:3_SI     | Third Person Singular (referring to someone else, singular)    | He is reading.                                      |
| PERSON:3_PL     | Third Person Plural (referring to a group of others)           | They are reading.                                   |
| TRANSITIV:ITR   | Intransitive (that do not require a direct object)             | She sleeps.                                         |
| STATUS:CMP      | Complete Action Status (an action that is finished)            | She has finished the project.                       |
| STATUS:ICM      | Incomplete Action Status (an action that is ongoing or not finished) | She is working on the project.                   |