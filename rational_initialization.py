import argparse
import json
import os
import random
import re
import time

from tqdm import tqdm

import sys
import os

from prompt_tools import get_formed_key_phase, get_prompt_for_initial_rational
from utils import prepare_llm, prepare_label_set


def prepare_prompts_for_explanations(args, labeled_samples):
    re_x_prompts = []
    for idx, labeled_sample in enumerate(tqdm(labeled_samples)):
        re_x_prompt = get_prompt_for_initial_rational(
            args=args,
            labeled_sample=labeled_sample,
        )
        re_x_prompts.append(re_x_prompt)
    return re_x_prompts


def parse_explanation_from_text(generated_text):
    parsed_explanations = generated_text

    # 第一步，找到<End of Instance>并只保留这之前的部分
    generated_text = generated_text.split('<End of Instance>')[0]

    if 'Explanations:' in generated_text:
        generated_text = generated_text.split('Explanations:')[1]

    if 'Given the sentence' in generated_text:
        parsed_explanations = generated_text.split('Given the sentence')[0]

    parsed_explanations = parsed_explanations.replace('\n', '')
    return parsed_explanations


def parse_key_phases_from_explanation(explanations):
    key_phases = None

    if 'phrase' not in explanations:
        pass
    else:
        phrase_sentence = explanations.split('phrase')[1]
        # print(phrase_sentence)

        matches = re.findall(r'\"(.*?)\"', phrase_sentence)

        if len(matches) > 0:
            key_phases = matches[0]
        else:
            key_phases = None

    return key_phases


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--k_shot', type=int, default=10, help='number of examples per class')
    parser.add_argument('--dataset', type=str, default='SemEval', help='name of the dataset')
    parser.add_argument('--llm_type', type=str, default='llama2_7b_chat', help='llm type ')
    args = parser.parse_args()

    ################################################################
    ######## 准备类别标签集合 #########################################
    relation_type_set, relation_set_dict = prepare_label_set(dataset=args.dataset)
    setattr(args, 'relation_type_set', relation_type_set)
    setattr(args, 'relation_set_dict', relation_set_dict)
    ################################################################

    with open(f'./data/{args.dataset}/sampled_{str(args.k_shot)}_shot_train.json') as f:
        labeled_samples = json.loads(f.read())



    response_record_dir = 'llm_generated_re_x'
    if not os.path.exists(response_record_dir):
        os.mkdir(response_record_dir)
    response_record_filepath = \
        f'{response_record_dir}/{args.dataset}_{str(args.k_shot)}_shot_response_record_{args.llm_type}_x.txt'
    response_record_filepath = open(response_record_filepath, encoding='utf-8', mode='w')

    prompts = prepare_prompts_for_explanations(args=args, labeled_samples=labeled_samples)

    llm, sampling_params = prepare_llm(args=args)

    # 使用 llm.generate 方法生成输出文本。
    # 这会将输入提示加入 vLLM 引擎的等待队列，并执行引擎以高效地生成输出
    outputs = llm.generate(prompts, sampling_params)

    demonstrations_with_x = []

    for output, labeled_sample in zip(outputs, labeled_samples):
        prompt = output.prompt  # 获取原始的输入提示
        generated_text = output.outputs[0].text  # 从输出对象中获取生成的文本
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        explanations = parse_explanation_from_text(generated_text=generated_text)
        formed_key_phase = get_formed_key_phase(head_entity=labeled_sample["head_entity"],
                                                tail_entity=labeled_sample["tail_entity"],
                                                given_sentence=labeled_sample["sentence"])
        explanations = formed_key_phase + explanations


        print('%%%%%%%%%%%%%%%%%%%%%% prompt %%%%%%%%%%%%%%%%%%%%%%', file=response_record_filepath)
        print(prompt, file=response_record_filepath)
        print('%%%%%%%%%%%%%%%%%%%%%% prompt %%%%%%%%%%%%%%%%%%%%%%', file=response_record_filepath)
        print(generated_text, file=response_record_filepath)

        print('########################', file=response_record_filepath)
        print(labeled_sample["relation_type"], file=response_record_filepath)
        print('########################', file=response_record_filepath)

        print('########################', file=response_record_filepath)
        print(explanations, file=response_record_filepath)
        print('########################', file=response_record_filepath)
        print('\n', file=response_record_filepath)

        labeled_sample['explanations'] = explanations
        labeled_sample['sentence_str'] = ' '.join(labeled_sample['sentence'])

        demonstrations_with_x.append(labeled_sample)

    demonstrations_with_x_file = f'./data/{args.dataset}/sampled_{str(args.k_shot)}_shot_train_with_{args.llm_type}_x.json'
    with open(demonstrations_with_x_file, encoding='utf-8', mode='w') as f:
        json.dump(demonstrations_with_x, f, indent=2)


