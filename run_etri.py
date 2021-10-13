import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

# 0828 추가
from season2_electra import ExtractedFeatElectraForSequenceClassification, AttentionFeatElectraForSequenceClassification, ResnetAttentionElectraForSequenceClassification

from etri_dataset import ETRIDataset, EtriProcessor
from etri_trainer import ETRITrainer
from etri_electra import ImageElectraForSequenceClassification, OnlyElectraForSequenceClassification, ResnetElectraForSequenceClassification, CLSResnetElectraForSequenceClassification, ResnetConcatElectraForSequenceClassification
# from etri_bert import ImageBertForSequenceClassification, OnlyBertForSequenceClassification

# Kobert 전용 토크나이저 import
from tokenization_kobert import KoBertTokenizer
from itertools import permutations

# 원래 GLUE Task에 없는 애들 추가
glue_tasks_num_labels['etri']=2
glue_output_modes['etri']='classification'
# glue_compute_metrics['etri'] = glue_compute_metrics['qnli']

#TODO: evaluate_rank에서는 prediction해서 파일을 생성한 후에, 따로 label.txt랑 평가하는 방식으로 해야함.
#TODO: 그래야 호환 가능
logger = logging.getLogger(__name__)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    # kobert와의 호환성
    if 'kobert' not in model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    else: # 모델 이름에 kobert가 들어있을 때
        tokenizer = KoBertTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    #TODO 임시방편 고쳐야함
    model = AttentionFeatElectraForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # num_added_toks = tokenizer.add_tokens(['_EOT_'])
    # model.resize_token_embeddings(len(tokenizer))

    # Get datasets
    train_dataset = (
        ETRIDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        ETRIDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    test_dataset = (
        ETRIDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
        if training_args.do_predict
        else None
    )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics('qnli', preds, p.label_ids)
            # return {"acc": simple_accuracy(preds, p.label_ids)}

        return compute_metrics_fn

    def save_predictions(predictions, guids):
        len_examples = len(set([guid.split('-')[1] for guid in guids]))
        result_array = np.zeros([len_examples, 3])
        # predictions_tensor = torch.tensor(predictions)
        # predictions = torch.nn.Softmax(dim=-1)(predictions_tensor).numpy()
        for pred, guid in zip(predictions, guids):
            example_id, response_num = int(guid.split('-')[1]), int(guid.split('-')[2])
            result_array[example_id][response_num-1] += pred[1] # True label의 값을 더해서 나중에 비교
        # results = np.argsort(result_array, axis=1)
        # np.argsort 2번 사용해서 결국 [0.8, 0.2, 0.5] => [2 0 1] 순서로 랭크값을 가지게 했음. 여기서 테이블로 변환시켜서 숫자로 치환해야함.
        # 설명해준거랑 거꾸로 [2, 1, 0], [2, 0, 1], [1, 2, 0], [1, 0, 2], [0, 2, 1], [0, 1, 2] 순서

        tmp = np.argsort(result_array, axis=1)
        results = np.argsort(tmp, axis=1)

        rank_table = np.array(list(permutations(np.arange(3), 3))[::-1])
        # output_test_file = '/home/work/model/prediction.csv' #TODO Hard Coding.
        output_test_file = os.path.join(
                training_args.output_dir, f"prediction.csv"
        )
        with open(output_test_file, 'w') as writer:
            for rank1, rank2, rank3 in results:
                for i, table in enumerate(rank_table):
                    if rank1 == table[0] and rank2 == table[1] and rank3 == table[2]:
                        writer.write("%s\n" % (i))


                # writer.write("%s %s %s\n" % (rank1, rank2, rank3))


    # Initialize our Trainer
    trainer = ETRITrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
            )

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
            )

        for test_dataset in test_datasets:
            _predictions, guids = trainer.predict(test_dataset=test_dataset)
            predictions = _predictions.predictions

            if trainer.is_world_master():
                save_predictions(predictions, guids)
            """
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
            """
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
