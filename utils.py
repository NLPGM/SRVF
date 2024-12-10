from tqdm import tqdm

from prompt_tools import ChatGPT_Class

import random

import numpy as np
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader


def prepare_label_set(dataset):
    # 注意，这里的label列表第一个（索引为0）设置为Other或者无关系类型，因为在解析不出来时会默认为此Other/no_relation
    if dataset in ['TACRED', 'TACRED_ZeroShot']:
        label_set = ['no_relation', 'org:founded_by', 'per:employee_of',
                     'org:alternate_names', 'per:cities_of_residence',
                     'per:children', 'per:title', 'per:siblings',
                     'per:religion', 'per:age', 'org:website',
                     'per:stateorprovinces_of_residence',
                     'org:member_of', 'org:top_members/employees',
                     'per:countries_of_residence', 'org:city_of_headquarters',
                     'org:members', 'org:country_of_headquarters',
                     'per:spouse', 'org:stateorprovince_of_headquarters',
                     'org:number_of_employees/members', 'org:parents',
                     'org:subsidiaries', 'per:origin', 'org:political/religious_affiliation',
                     'per:other_family', 'per:stateorprovince_of_birth',
                     'org:dissolved', 'per:date_of_death', 'org:shareholders',
                     'per:alternate_names', 'per:parents', 'per:schools_attended',
                     'per:cause_of_death', 'per:city_of_death', 'per:stateorprovince_of_death',
                     'org:founded', 'per:country_of_birth', 'per:date_of_birth',
                     'per:city_of_birth', 'per:charges', 'per:country_of_death']
    elif dataset in ['SemEval', 'SemEval_ZeroShot']:
        label_set = ['Other',
                     'Component-Whole',
                     'Instrument-Agency',
                     'Member-Collection',
                     'Cause-Effect',
                     'Entity-Destination',
                     'Content-Container',
                     'Message-Topic',
                     'Product-Producer',
                     'Entity-Origin']
    elif dataset in ['Re-TACRED']:
        label_set = ["no_relation",
                     "per:cause_of_death", "org:city_of_branch", "per:schools_attended",
                     "org:founded", "org:dissolved", "per:origin",
                     "org:number_of_employees/members", "per:city_of_birth", "per:country_of_birth",
                     "per:stateorprovince_of_birth", "per:other_family", "per:religion",
                     "per:spouse", "per:age", "per:siblings",
                     "org:political/religious_affiliation", "org:country_of_branch", "per:charges",
                     "per:date_of_death", "per:cities_of_residence", "org:member_of",
                     "per:parents", "org:alternate_names", "org:members",
                     "org:shareholders", "per:employee_of", "org:website",
                     "per:identity", "per:stateorprovince_of_death", "per:children",
                     "per:stateorprovinces_of_residence", "org:stateorprovince_of_branch", "per:city_of_death",
                     "per:date_of_birth", "per:countries_of_residence", "per:title",
                     "org:founded_by", "per:country_of_death", "org:top_members/employees"]
    elif dataset in ['DOCRED']:
        label_set = [
            'employer', 'capital', 'capital of', 'founded by', 'basin country', 'work location', 'parent taxon',
            'end time', 'follows', 'head of government', 'country', 'headquarters location', 'applies to jurisdiction',
            'publication date', 'ethnic group', 'series', 'date of death', 'residence', 'start time', 'participant of',
            'characters', 'award received', 'replaced by', 'subclass of', 'legislative body', 'father', 'spouse',
            'performer', 'official language', 'operator', 'production company', 'director', 'notable work',
            'unemployment rate', 'creator', 'sister city', 'military branch', 'lyrics by', 'platform',
            'original language of work', 'participant', 'influenced by', 'languages spoken, written or signed',
            'continent', 'product or material produced', 'part of', 'chairperson', 'country of origin',
            'member of political party', 'location', 'head of state', 'dissolved, abolished or demolished',
            'manufacturer', 'developer', 'followed by', 'owned by', 'territory claimed by', 'position held',
            'conflict', 'located on terrain feature', 'member of sports team', 'mouth of the watercourse',
            'location of formation', 'inception', 'author', 'screenwriter', 'instance of', 'subsidiary', 'league',
            'composer', 'located in or next to body of water', 'parent organization', 'point in time', 'producer',
            'place of death', 'sibling', 'narrative location', 'date of birth',
            'located in the administrative territorial entity', 'present in work', 'religion', 'separated from',
            'contains administrative territorial entity', 'has part', 'replaces', 'genre', 'educated at',
            'country of citizenship', 'place of birth', 'publisher', 'record label', 'child', 'mother', 'cast member',
            'original network', 'member of'
        ]

    label_dict = {}
    for idx, item in enumerate(label_set):
        label_dict[item] = idx

    return label_set, label_dict


def prepare_llm(args, temperature=0, n=1):
    if args.llm_type == "gpt-3.5-turbo":
        sampling_params = prepare_sampling_params(args=args, temperature=temperature, n=n)
        llm = ChatGPT_Class(llm_type="gpt-3.5-turbo-0125")

    else:
        from vllm import LLM

        # Your own path for the llms.
        path = "/data01/qty/liyongqi/llm_path/meta-llama"

        # 定义采样参数，temperature 控制生成文本的多样性，top_p 控制核心采样的概率
        sampling_params = prepare_sampling_params(args=args, temperature=temperature, n=n)

        if "70" in args.llm_type:
            gpu_memory_utilization = 0.9
            llm = LLM(model=f"{path}/{args.llm_type}",
                      gpu_memory_utilization=gpu_memory_utilization,
                      )
        else:
            gpu_memory_utilization = 0.9
            llm = LLM(model=f"{path}/{args.llm_type}",
                      gpu_memory_utilization=gpu_memory_utilization)

    return llm, sampling_params


def prepare_sampling_params(args, temperature, n):
    if args.llm_type == 'gpt-3.5-turbo':
        sampling_params = {
            "temperature": temperature,
            "n": n
        }
    else:
        from vllm import SamplingParams

        sampling_params = SamplingParams(temperature=temperature,
                                             top_p=1,
                                             max_tokens=512,
                                             n=n,
                                             stop=["<End of Instance>"],
                                             include_stop_str_in_output=True)


    return sampling_params


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def GetDataLoader(args, samples, batch_size):
    features = []
    for sample in tqdm(samples, total=len(samples), desc="Loading data"):
        if 'rational' in sample.keys():
            cat_label = sample["cat_label"]
            cat_label_id = args.cat_labels_dict[cat_label]
            rational = sample["rational"]

            features.append(convert_to_feature_rational(args, cat_label_id, rational))
        else:
            feature = convert_to_feature(args, sample)
            if feature is not None:
                features.append(feature)
    dataset = convert_features_to_dataset(features)
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset,
                                  sampler=train_sampler,
                                  batch_size=batch_size)
    return train_dataloader


def GetDataEnumerater(args, samples, batch_size):
    features = []
    for sample in samples:
        if 'rational' in sample.keys():
            cat_label_id = 0  # GetDataEnumerater只用于测试阶段，所以这里只需要随意设一个id
            rational = sample["rational"]
            features.append(convert_to_feature_rational(args, cat_label_id, rational))
        else:
            feature = convert_to_feature(args, sample)
            if feature is not None:
                features.append(feature)

    dataset = convert_features_to_dataset(features)
    train_dataloader = DataLoader(dataset,
                                  batch_size=batch_size)
    return train_dataloader


def convert_to_feature_rational(args, cat_label_id, rational):
    max_seq_length = args.max_seq_length

    sample_tokens = args.tokenizer.tokenize(rational)
    sample_tokens = ["[CLS]"] + sample_tokens + ["[SEP]"]

    input_ids = args.tokenizer.convert_tokens_to_ids(sample_tokens)
    padding_length = max_seq_length - len(input_ids)

    if padding_length >= 0:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids_padded = input_ids + [0] * padding_length

    else:
        attention_mask = ([1] * len(input_ids))[:max_seq_length]
        input_ids_padded = input_ids[:max_seq_length]

    special_mask_padded = [0] * max_seq_length
    # 将cls位置置为1，以便后续获取该句子的向量
    special_mask_padded[0] = 1

    token_type_ids = [0] * max_seq_length

    assert len(input_ids_padded) == max_seq_length
    assert len(special_mask_padded) == max_seq_length

    assert len(token_type_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length

    return InputFeature(input_ids_padded, special_mask_padded, token_type_ids, attention_mask,
                        label_id=cat_label_id)


def convert_to_feature(args, sample):
    max_seq_length = args.max_seq_length

    left_start_idx = sample["head_entity"]["start_idx"]
    left_end_idx = sample["head_entity"]["end_idx"]

    right_start_idx = sample["tail_entity"]["start_idx"]
    right_end_idx = sample["tail_entity"]["end_idx"]

    if left_start_idx > right_start_idx:
        tmp_left_start_idx = left_start_idx
        tmp_left_end_idx = left_end_idx

        tmp_right_start_idx = right_start_idx
        tmp_right_end_idx = right_end_idx

        left_start_idx = tmp_right_start_idx
        left_end_idx = tmp_right_end_idx
        right_start_idx = tmp_left_start_idx
        right_end_idx = tmp_left_end_idx

    words = sample["sentence"]

    words_after_special_mask = []
    words_special_mask = []
    for idx, word in enumerate(words):
        if idx == left_start_idx:
            words_after_special_mask.append("$")
            words_after_special_mask.append(word)
            words_special_mask.append(2)
            words_special_mask.append(4)
        elif idx == right_start_idx:
            if idx == left_end_idx + 1:
                # 专门处理 左右实体相邻的问题
                words_after_special_mask.append("$")
                words_special_mask.append(2)

            words_after_special_mask.append("#")
            words_after_special_mask.append(word)
            words_special_mask.append(3)
            words_special_mask.append(5)

        elif idx == left_end_idx + 1:
            words_after_special_mask.append("$")
            words_after_special_mask.append(word)
            # words_special_mask.append(2)
            words_special_mask.append(7)

            words_special_mask.append(6)

        elif idx == right_end_idx + 1:
            words_after_special_mask.append("#")
            words_after_special_mask.append(word)
            # words_special_mask.append(3)
            words_special_mask.append(8)

            words_special_mask.append(6)
        else:
            words_after_special_mask.append(word)
            words_special_mask.append(6)

    if right_end_idx == len(words) - 1:
        # 特殊情况
        words_after_special_mask.append("#")
        # words_special_mask.append(3)
        words_special_mask.append(8)

    words_after_special_mask = ["[CLS]"] + words_after_special_mask + ["[SEP]"]
    words_special_mask = [1] + words_special_mask + [0]

    sample_tokens = []
    special_mask = []

    for word, special_mask_id in zip(words_after_special_mask, words_special_mask):
        word_tokens = args.tokenizer.tokenize(word)
        if len(word_tokens) == 0:  # Meet special space character
            word_tokens = args.tokenizer.tokenize('[UNK]')
            word_special_mask_tokens = [0]
        else:
            word_special_mask_tokens = [special_mask_id] + [6] * (len(word_tokens) - 1)

        sample_tokens.extend(word_tokens)
        special_mask.extend(word_special_mask_tokens)

    input_ids = args.tokenizer.convert_tokens_to_ids(sample_tokens)
    padding_length = max_seq_length - len(input_ids)

    if padding_length >= 0:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids_padded = input_ids + [0] * padding_length
        special_mask_padded = special_mask + [0] * padding_length

    else:
        attention_mask = ([1] * len(input_ids))[:max_seq_length]
        input_ids_padded = input_ids[:max_seq_length]
        special_mask_padded = special_mask[:max_seq_length]

    token_type_ids = [0] * max_seq_length

    if 4 not in special_mask_padded or 5 not in special_mask_padded:
        print(sample_tokens)
        print(special_mask_padded)
        print(sample)
        print(len(sample_tokens))
        print('Entity marker loses, because of length.')

        # assert 1 == 0
        return None

    assert len(input_ids_padded) == max_seq_length
    assert len(special_mask_padded) == max_seq_length

    assert len(token_type_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length

    return InputFeature(input_ids_padded, special_mask_padded, token_type_ids, attention_mask,
                        label_id=args.relation_set_dict[sample["relation_type"]])


class InputFeature(object):
    def __init__(self, input_ids, special_mask, token_type_ids, attention_mask, label_id):
        self.input_ids = input_ids
        self.special_mask = special_mask
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.label_id = label_id


def convert_features_to_dataset(features):
    # convert to Tensors
    all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
    all_special_mask = torch.tensor([feature.special_mask for feature in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([feature.token_type_ids for feature in features], dtype=torch.long)
    all_attention_mask = torch.tensor([feature.attention_mask for feature in features], dtype=torch.long)
    all_labels_id = torch.tensor([feature.label_id for feature in features])
    dataset = TensorDataset(all_input_ids, all_special_mask, all_token_type_ids, all_attention_mask, all_labels_id)
    return dataset


def filter_samples_over_length(args, samples):
    max_seq_length = args.max_seq_length

    filtered_samples = []
    for sample in samples:
        tokens = args.tokenizer.tokenize(' '.join(sample["sentence"]))
        if len(tokens) + 3 < max_seq_length:
            filtered_samples.append(sample)

    return filtered_samples
