from transformers.data.processors.glue import *
from transformers.data.datasets.glue import *

from PIL import Image
from torchvision import transforms

from tqdm import tqdm

# ETRI에서 제공하는 feature 다운받기 위해서 사용
import json
import numpy as np

@dataclass
class EtriExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    image_name: Optional[str] = None # 여기에도 이미지 이름을 저장
    label: Optional[str] = None
    description_length: Optional[int] = 0

@dataclass
class EtriFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
        image: (Oprtional) Image file of the corresponding cloth ID.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    image_name: Optional[List[str]] = None # image 이름을 가지고 실시간으로 불러오기 위함
    description_length: Optional[int] = None



class EtriProcessor(DataProcessor):
    """ Processor for the ETRI data set."""

    def train_to_examples(self, input_file, mdata, qutochar='\t'):
        """
        train.txt는 id \t sentence1 \t 옷코드 \t label 형식으로 이루어져 있음.
        """
        examples = []
        with open(input_file, 'r', encoding='utf-8') as f: # 임시로 이렇게 해놓음 나중에 encoding 생각
            for i, line in enumerate(tqdm(f.readlines())):
                # 첫 줄에 column 정보 있을 때 이 부분 수정
                # if i == 0:
                #     continue

                # ID 있는 버전
                # id, dialogue, clothID, label = line.strip().split('\t')
                # guid = '%s-%s-%s' % ('train', id, 0) # Dev와 꼴을 맞춰주기 위함.

                # ID 없는 버전
                dialogue, clothID, label = line.strip().split('\t')
                guid = '%s-%s-%s' % ('train', i, 0) # Dev와 꼴을 맞춰주기 위함.

                text_a = dialogue.strip()
                text_b = ' '.join(mdata[clothID][0])
                image_name = clothID
                examples.append(EtriExample(guid=guid, text_a=text_a, text_b=text_b, image_name=image_name, label=label))
        return examples


    def read_etri(self, input_file, qutochar='\t'):
        with open(input_file, 'r', encoding='EUC-KR') as f:

            dev_data = [] # 전체 데이터

            dialogue = [] # 데이터의 대화 부분

            r1 = [] # 일반적으로 r1이 정답(dev에서는)
            r2 = [] # 그다음에 얘
            r3 = [] # 마지막

            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                if line[0] == ';':
                    if dialogue:
                        dev_data.append((' '.join(dialogue), r1, r2, r3))
                        dialogue = []
                        r1, r2, r3 = [], [], []
                else:
                    speaker, content = line.split('\t')
                    if speaker == 'US' or speaker == 'CO':
                        dialogue.append(content)
                    else:
                        if speaker == 'R1':
                            r1 = content.strip().split()
                        elif speaker == 'R2':
                            r2 = content.strip().split()
                        elif speaker == 'R3':
                            r3 = content.strip().split()

            return dev_data

    def read_mdata(self, input_file):
        """ dictionary 형태로 코드번호: Tuple(Description(list), 대분류, 소분류) 형태"""
        #TODO: Description 내용을 기준에 따라 분리할 것인지 아닌지 정해야함
        f = open(input_file, encoding='CP949')
        dic = {}
        for line in f.readlines():
            ids, large, small, _, description = line.strip().split('\t')
            if ids.strip() not in dic:
                dic[ids.strip()] = ([description], large, small)
            else:
                dic[ids.strip()][0].append(description)

        return dic
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.txt")), "train")
        return self.train_to_examples(os.path.join(data_dir, "train.txt"),\
            self.read_mdata(os.path.join(data_dir, "mdata.txt.2020.6.23.txt")))

    # 이 부분 고쳤음
    def get_dev_examples(self, data_dir):
         return self._create_examples(lines=self.read_etri(os.path.join(data_dir, "ac_eval_t1.dev.txt")),\
            mdata=self.read_mdata(os.path.join(data_dir, "mdata.txt.2020.6.23.txt")))

    # def get_dev_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(lines=self.read_etri(os.path.join(data_dir, "ac_eval_t1.dev.txt")),\
            mdata=self.read_mdata(os.path.join(data_dir, "mdata.txt.2020.6.23.txt")))

    def get_labels(self):
        """See base class."""
        return ["0", "1"] # 1이 긍정 0이 부정

    def _create_examples(self, lines, mdata, set_type='test'):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(tqdm(lines)):
            dialogue, r1, r2, r3 = line
            for r_idx, response in enumerate((r1, r2, r3)):
                for clothID in response:
                    guid = '%s-%s-%s' % (set_type, i, str(r_idx+1))
                    text_a = dialogue.strip()
                    text_b = ' '.join(mdata[clothID][0])
                    image_name = clothID
                    # image = Image.open('/workspace/data/image/'+clothID+'.jpg') #TODO 나중에 꼭 수정!!!!!!!!!!!!!! 현재 하드코딩
                    label = None
                    examples.append(EtriExample(guid=guid, text_a=text_a, image_name=image_name, text_b=text_b, label=label))
            # splited = line.split('\t')
            # guid = "%s-%s" % (set_type, splited[0])
            # text_a = splited[1]
            # text_b = splited[2]
            # label = None if set_type == "test" else None
            # examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def etri_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = EtriProcessor()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]
    tokenizer.pad_token = '<pad>'
    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    # 여기서 안하고 train time 때 online으로 할 것임
    # image_transform = transforms.Compose([
    #     transforms.Resize((100, 100)), \
    #     transforms.ToTensor() \
    #     ])
    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        inputs['image_name'] = examples[i].image_name
        # inputs['image'] = image_transform(examples[i].image) # image 데이터 추가.
        feature = EtriFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features, [example.guid for example in examples]

class ETRIDataset(Dataset):
    """ Dataset for ETRI Data."""
    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = EtriProcessor() # EtriProcessor 고정
        self.output_mode = "classification" # Classification 고정

        # ETRI에서 제공하는 [3, 4096] feature 사용하기 위해 이용.
        # 파일 이름으로 key, [3, 4096] feature를 value로 가지고 있는 dictionary
        self.extracted_feat = None
        with open(os.path.join(self.args.data_dir, 'extracted_feat.json'), 'r') as f:
            self.extracted_feat = json.load(f)

        # Pretrained model 전용 normalize and preprocess
        self.transforms = transforms.Compose([
            transforms.Resize((100, 100)),
            # transforms.CenterCrop(224), # 기존 Resnet 전용
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        label_list = self.processor.get_labels()
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        #TODO 하드코딩
        # lock_path = cached_features_file + ".lock"
        # with FileLock(lock_path):

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            start = time.time()
            (self.features, self.guids) = torch.load(cached_features_file)
            logger.info(
                f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
            )
        else:
            logger.info(f"Creating features from dataset file at {args.data_dir}")

            if mode == Split.dev:
                examples = self.processor.get_dev_examples(args.data_dir)
            elif mode == Split.test:
                examples = self.processor.get_test_examples(args.data_dir)
            else:
                examples = self.processor.get_train_examples(args.data_dir)
            if limit_length is not None:
                examples = examples[:limit_length]
            self.features, self.guids = etri_convert_examples_to_features(
                examples,
                tokenizer,
                max_length=args.max_seq_length,
                label_list=label_list,
                output_mode=self.output_mode,
            )
            start = time.time()
            # torch.save((self.features, self.guids), cached_features_file)
            # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
            logger.info(
                "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> EtriFeatures:
        image = Image.open(os.path.join(self.args.data_dir, 'image', self.features[i].image_name+'.jpg'), 'r')
        image.load()

        # extracted_feat은 numpy 형태로 되어있으므로 tensor형태로 변환 후 넣는다.
        extracted_feat = torch.tensor(np.array(self.extracted_feat[self.features[i].image_name+'.jpg']), dtype=torch.float32)

        # dictionary 형태로 feature 변환, 이후 병합 후 return

        
        # return {**self.features[i].__dict__, 'image': self.transforms(image)}

        # 위쪽께 원본, 이전 모델들은 위 방식대로 안하면 model output출력할때 오류뜸
        return {**self.features[i].__dict__, 'image': None, 'extracted_feat': extracted_feat}

        # return {**self.features[i].__dict__}

    def get_labels(self):
        return self.label_list

    def get_guids(self):
        return self.guids
