from mmengine.config import read_base

with read_base():
    from .datasets.subjective.multiround.mtbench101_judge import subjective_datasets

from opencompass.models import OpenAI, FragmentHF
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import MTBench101Summarizer

# ---------------------------------------------------------------------------------------------------------

api_meta_template = dict(
    round=[
        dict(role='SYSTEM', api_role='SYSTEM'),
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models
models = [
    dict(
        type=FragmentHF,
        abbr='llama3.1_fragment',
        hf_key='',
        path = 'meta-llama/Llama-3.2-1B-Instruct',
        search_model_name = 'sentence-transformers/all-mpnet-base-v2',
        original_data_path = None,
        max_out_len=4096,
        batch_size=1,
    )
]

datasets = [*subjective_datasets]

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=10000),
    runner=dict(
        type=LocalRunner,
        partition='llm_dev2',
        quotatype='auto',
        max_num_workers=1,
        task=dict(type=OpenICLInferTask),
    ),
)

# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration
judge_models = [dict(
    abbr='GPT4-Turbo',
    type=OpenAI,
    path='gpt-4o-mini', # To compare with the official leaderboard, please use gpt-4-1106-preview
    key='',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
    meta_template=api_meta_template,
    query_per_second=16,
    max_out_len=4096,
    max_seq_len=4096,
    batch_size=8,
    temperature=0.8,
)]

## ------------- Evaluation Configuration



eval = dict(
    partitioner=dict(type=SubjectiveSizePartitioner, max_task_size=100000, mode='singlescore', models=models, judge_models=judge_models),
    runner=dict(type=LocalRunner, max_num_workers=32, task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=MTBench101Summarizer, judge_type='single')

work_dir = 'outputs/mtbench101/'
