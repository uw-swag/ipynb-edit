# ipynb-edit

This repo contains the code and data for our FSE-IVR 2025 paper: [Learning to Edit Interactive Machine Learning Notebooks](https://pengyunie.github.io/p/JinETAL25NotebookEdit.pdf).

> Machine learning (ML) developers frequently use interactive computational notebooks, such as Jupyter notebooks, to host code for data processing and model training. Notebooks provide a convenient tool for writing ML pipelines and interactively observing outputs. However, maintaining notebooks, e.g., to add new features or fix bugs, can be challenging due to the length and complexity of the ML pipeline code. Moreover, there is no existing benchmark related to developer edits on notebooks.
> In this paper, we present early results of the first study on learning to edit ML pipeline code in notebooks using large language models (LLMs). We collect the first dataset of 48,398 notebook edits derived from 20,095 revisions of 792 ML-related GitHub repositories. Our dataset captures granular details of file-level and cell-level modifications, offering a foundation for understanding real-world maintenance patterns in ML pipelines. We observe that the edits on notebooks are highly localized. Although LLMs have been shown to be effective on general-purpose code generation and editing, our results reveal that the same LLMs, even after finetuning, have low accuracy on notebook editing, demonstrating the complexity of real-world ML pipeline maintenance tasks. Our findings emphasize the critical role of contextual information in improving model performance and point toward promising avenues for advancing LLMs' capabilities in engineering ML code.

If you use our dataset or build on our work in your research, please cite:
```bibtex
@inproceedings{JinETAL25NotebookEdit,
  title={Learning to Edit Interactive Machine Learning Notebooks},
  author={Jin, Bihui and Wang, Jiayue and Nie, Pengyu},
  booktitle={Proceedings of the Symposium on the Foundations of Software Engineering, Ideas, Visions, and Reflections Track},
  year={2025},
}
```

## Data

Our dataset of notebook edits are available on [Zenodo](https://doi.org/10.5281/zenodo.15716537)
- commits.jsonl: Our dataset of Jupyter notebook edits collected from ML-related GitHub repositories.
- commits-filtered.jsonl: The same dataset after filtering (removing edits with < 3 words of commit messages), suitable for LLM learning experiments.


## Code: Dataset Collection and Processing

You can replicate our experiment following the below procedures:

### Data collection:

``python_fetch.py`` uses GitHub API to collect repos and will output ``top_1000_python_repos.json`` (which contains top 1000 repos sorted by popularity).

``change_stat.py`` clones the repos from ``top_1000_python_repos.json`` and output ``commits.jsonl``, which contains all the data used in analysis (e.g., repo name, commit message, commit hash, file name, old content, and new content) on each line for a file-level edit

### Data processing:

``split.py`` removes the duplicate in ``commits.jsonl`` and splits into test (``test_index.txt``), training (``train_index.txt``) and validation (``val_index.txt``) sets (by outputing the corresponding index).

``commits.jsonl`` is then saved again to remove duplications.


## Code: LLM Learning Experiments

### Finetuning

``train_baseline.py`` uses LoRA to finetune the model.

To control the inference types (cell/file) and the sequence, change the following parameter:

```

    model_dir = '1.3b_file' # used to infer file edits with 1.3b model. '1.3b_cell' infers cell-level edits with 1.3b model.

    cycle = '3' # 1,2,3 - the (n)th model

```

and the script (with above params) will yield a finetuned model in the folder ``1.3b_file/final``.

``lora_config.json`` is defined as follows:

where ``tokenizer.path`` is the model name used for the tokenizer setting (change to either deepseek-ai/deepseek-coder-1.3b-instruct or deepseek-ai/deepseek-coder-6.7b-instruct),

``model.path`` is the model name used for the model setting (change to either deepseek-ai/deepseek-coder-1.3b-instruct or deepseek-ai/deepseek-coder-6.7b-instruct),

``model.rank``, ``alpha``, and ``learning_rate`` need to be set based on your need, 

and ``training.epochs``, ``per_device_train_batch_size``, ``per_device_test_batch_size``, and ``gradient_accumulation_steps`` need to be set based on your need.

Note that we recommand that ``effective_batch_size`` = ``per_device_train_batch_size`` (as large as possible) *  ``gradient_accumulation_steps`` * device count (1 if you're using 1 GPU) should be at least 64 (that's the value we used), but can also be larger (I've seen as much as 512).

``train_baseline.py`` uses functions ``from .common import build_tokenizer, build_trainer``.

### Baseline construction and inference:

``baseline.py`` infers the code edits based on the user input.

You need to change the following parameters:

To use the basic model (unfinetuned), set:

```

    ftt=0

    finetune = file[ftt] (the 6.7B model) or cell[ftt] (the 1.3B model)

```

To load the (i)th finetuned model, set:

```

    ftt=i (i=1,2,3)

    finetune = file[ftt] (the 6.7B model) or cell[ftt] (the 1.3B model)

```

To infer cell-level edits with zero exemplar, run:

```

    print("Cell diff zero shot")

    for d in tqdm(test):

    name = d.file.split("/")[-1]

    file_name = f"model/results/cell_only_zero_shot_1.3b_{ftt}/" + d.repo + "_" + d.commit + "_" + name + ".txt"

    torch.cuda.empty_cache()

    if not os.path.isfile(file_name):

    tgt = generate_code_change_cell_only(tokenizer, model, d.message, d.old, d.cell_diff)

    with open(file_name, "w") as text_file:

    text_file.write(tgt)

```

To infer file-level edits with zero exemplar, run:

```

    print("Whole file one shot")

    for d in tqdm(test):

    name = d.file.split("/")[-1]

    file_name = f"model/results/whole_file_zero_shot_1.3b_{ftt}/" + d.repo + "_" + d.commit + "_" + name + ".txt"

    torch.cuda.empty_cache()

    if not os.path.isfile(file_name):

    tgt = generate_code_change_cell_only(tokenizer, model, d.message, d.old, d.cell_diff)

    with open(file_name, "w") as text_file:

    text_file.write(tgt)

```

Similarly, one-shot and five-shot inference can be performed using the same approach. This involves providing one or five examples as context before the task prompt, following the same pattern to guide the model effectively during inference.

To infer code edits with the unfinetuned model, you may consider to use the below prompt in the ``generate_code_change_cell_only`` or ``generate_code_change_whole_file`` method (based on your inference types) to keep consistence.

```
src = f"""

    You are a skilled software developer with immense knowledge in software analysis and debugging.

    For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

    Your job is to generate Jupyter notebook python code changes given a commit message and original python code.

    ### Instruction:

    [

    Commit Message: "{commit_message}"

    Original Code:

    '''

    {original_code_prompt}

    '''

    ]

    It is your turn to response code in a cell format.

    ### Response:

    """
```

To infer code edits with the finetuned model, you may consider to use the below prompt in the ``generate_code_change_cell_only`` or ``generate_code_change_whole_file`` method to keep consistence.

```

    result = ""

    result += "### Instruction\n"

    result += "[\n"

    result += "Commit Message:" + "\" " + commit_message + "\"\n\n"

    result += "Original Code Cells:\n" 

    result += "\'\'\'\n"

    result += original_code_prompt + "\n"

    result += "\'\'\'\n"

    result += "]\n\n"

    result += "### Response:\n"

```

### Statistics of dataset

Run ``model_stat.py`` first then ``output_result.py`` to compute statistics of the dataset.

Change the following parameters based on your need in ``model_stat.py``:

```

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")

```

Run ``dataset_size_stat.py`` to get the size of the dataset (full, train, test, and val)

### evaluation metric calculation:

Run ``accuracy.py`` to calculate evaluation metrics (BLUE, CodeBLUE, EditSim, RougeL).

The script uses the site package ``teco``, where can be obtained from https://github.com/EngineeringSoftware/teco

You need to change the following parameters:

To calculate evaluation metrics of the basic model (unfinetuned), set:

```

    ftt=0

    finetune = file[ftt] (the 6.7B model) or cell[ftt] (the 1.3B model)

```

To load the (i)th finetuned model, set:

```

    ftt=i (i=1,2,3)

    finetune = file[ftt] (the 6.7B model) or cell[ftt] (the 1.3B model)

```

Change the output folder (inference) and expected folder (gold) directions:

```

    output_folder = "model/results/whole_file_zero_shot_1.3b" #cell_only_zero_shot_1.3b whole_file_five_shot_1.3b

    expected_folder = "model/results/expected_whole_file"     #expected_cell_only

```

### evaluation metric calculation for finetuned models:

  After running ``accuracy.py`` for 3 finetuned models, scores for 2 models will be stored as .pkl files.

  Run ``finetune_score.py`` to compute the average evaluation metrics.

  Change ``tpe = 'whole'`` for file-level edits and ``tpe = 'cell'`` for cell-level edits.
