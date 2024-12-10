import argparse
import copy
import json
import os
import random
import re
import time

from tqdm import tqdm

import sys
import os

from utils import prepare_label_set, prepare_llm
from prompt_tools import construct_original_scm_prompt, \
    construct_hard_intervention_prompt, construct_check_intervention_prompt, parse_original_scm_output, \
    parse_hard_intervention_output, parse_soft_intervention_output, get_formed_key_phase

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--k_shot', type=int, default=10, help='number of examples per class')
    parser.add_argument('--dataset', type=str, default='SemEval', help='name of the dataset')
    parser.add_argument('--llm_type', type=str, default='llama2_7b_chat', help='llm type ')

    args = parser.parse_args()

    # 准备带解释的标注样本集
    labeled_samples_with_x_filepath = f'./data/{args.dataset}/sampled_{str(args.k_shot)}_shot_train_with_{args.llm_type}_x.json'
    with open(labeled_samples_with_x_filepath, mode='r', encoding='utf-8') as f:
        labeled_samples_with_x = json.loads(f.read())


    # 准备类别标签集合
    relation_type_set, relation_set_dict = prepare_label_set(dataset=args.dataset)
    setattr(args, 'relation_type_set', relation_type_set)
    setattr(args, 'relation_set_dict', relation_set_dict)

    ###########################################################################
    # 以下设置均为因为开销考量，并不会对实验结果造成巨大干扰 (其中 num_select_other_id 小于类别数是确定的)
    if args.dataset in ['SemEval']:
        num_select_other_id = 8
    elif args.dataset in ['TACRED']:
        num_select_other_id = 20
    elif args.dataset in ['Re-TACRED']:
        num_select_other_id = 20

    if args.llm_type in ['gpt-3.5-turbo',
                         'llama2_70b_chat','llama2_70b_chat-GPTQ',
                         'Meta-Llama-3-70B-Instruct','Meta-Llama-3-70B-Instruct-GPTQ']:
        num_select_other_id = 4


    ###########################################################################
    # num_select_other_id = 2
    ###########################################################################

    # 准备记录文件
    response_record_dir = 'llm_generated_do_intervention'
    if not os.path.exists(response_record_dir):
        os.mkdir(response_record_dir)
    response_record_filepath = \
        f'{response_record_dir}/{args.dataset}_{str(args.k_shot)}_shot_record_{args.llm_type}_do_intervention.txt'
    response_record_filepath = open(response_record_filepath, encoding='utf-8', mode='w')

    # 准备llm
    llm, sampling_params = prepare_llm(args=args)

    # 统计标注样本的标签分布
    label_samples_dict = {}
    # 给标注样本附上独特的id；如：“l-1”表示“标注样本1”；
    for item_idx, item in tqdm(enumerate(labeled_samples_with_x), total=len(labeled_samples_with_x)):
        item_sample_id = f'l-{item_idx}'
        item['sample_id'] = item_sample_id
        item_label = item['relation_type']
        if item_label not in label_samples_dict:
            label_samples_dict[item_label] = [item_sample_id]
        else:
            label_samples_dict[item_label].append(item_sample_id)

    # for label in args.relation_type_set:
    #     if label not in label_samples_dict.keys():
    #         label_samples_dict[label]=[]

    # 做一个反向的sample_id到sample的映射表
    labeled_samples_with_x_dict = {}
    # 给每个标注样本新增一个属性，id_list_with_same_label；表示具有同标签，但是不是同一个样本的列表。
    for item_idx, item in tqdm(enumerate(labeled_samples_with_x), total=len(labeled_samples_with_x)):
        item_sample_id = item['sample_id']
        item_label = item['relation_type']
        id_list_with_same_label = label_samples_dict[item_label]
        tmp_id_list_with_same_label = copy.deepcopy(id_list_with_same_label)
        # 去掉自己的id
        tmp_id_list_with_same_label.remove(item_sample_id)

        item['id_list_with_same_label'] = tmp_id_list_with_same_label

        labeled_samples_with_x_dict[item_sample_id] = item

    # 接下来，对每个样本进行original_scm的生成、do_hard_intervention、do_soft_intervention（及检查）；
    # 并把最终的paired explanations保存至特定文件；
    # 注意pair-explanation的关键信息保留，头尾实体、关系类型、原句子都要做保留
    # 过程中保留生成的过程；

    selected_icl_list = []
    original_scm_prompt_list = []
    hard_intervention_prompt_list = []



    for item_idx, labeled_sample in tqdm(enumerate(labeled_samples_with_x),
                                         desc='Prepare prompts for original_scm and hard_intervention',
                                         total=len(labeled_samples_with_x)):
        item_id = f'l-{item_idx}'

        id_list_with_same_label = labeled_sample['id_list_with_same_label']

        tmp_relation_type_set=copy.deepcopy(args.relation_type_set)
        tmp_relation_type_set.remove(labeled_sample["relation_type"])
        selected_other_labels_relation_type_set = copy.deepcopy(random.sample(tmp_relation_type_set, k=num_select_other_id))
        selected_id_list=[]
        for other_label in selected_other_labels_relation_type_set:
            selected_id_list.append(random.choice(label_samples_dict[other_label]))

        # other_id_list = [f'l-{i}' for i in range(len(labeled_samples_with_x))]
        # other_id_list.remove(item_id)
        # selected_id_list = random.sample(other_id_list, k=num_select_other_id)

        for selected_id in selected_id_list:
            icl_sample = labeled_samples_with_x_dict[selected_id]
            selected_icl_list.append(icl_sample)

            original_scm_prompt = construct_original_scm_prompt(args=args,
                                                                sample=labeled_sample,
                                                                icl_samples=[icl_sample])

            hard_intervention_prompt = construct_hard_intervention_prompt(args=args,
                                                                          sample=labeled_sample,
                                                                          icl_samples=[icl_sample])

            original_scm_prompt_list.append(original_scm_prompt)
            hard_intervention_prompt_list.append(hard_intervention_prompt)

    all_prompts = original_scm_prompt_list + hard_intervention_prompt_list

    all_outputs = llm.generate(all_prompts, sampling_params)
    # 计算每个部分的长度
    part_length = len(original_scm_prompt_list)
    # 分割列表到三个部分
    original_scm_outputs = all_outputs[:part_length]
    hard_intervention_outputs = all_outputs[part_length:]

    original_scm_list = []
    hard_intervention_x_list = []

    check_original_scm_prompt_list = []
    check_hard_intervention_prompt_list = []

    repeated_labeled_samples_with_x = []
    for item in labeled_samples_with_x:
        repeated_labeled_samples_with_x.extend([item] * num_select_other_id)

    for item_idx, (labeled_sample,
                   selected_icl,
                   original_scm_output,
                   hard_intervention_output) in enumerate(zip(repeated_labeled_samples_with_x,
                                                              selected_icl_list,
                                                              original_scm_outputs,
                                                              hard_intervention_outputs)):
        # sample_label = labeled_sample['relation_type']

        original_scm_prompt = original_scm_output.prompt  # 获取原始的输入提示
        original_scm_generated_text = original_scm_output.outputs[0].text  # 从输出对象中获取生成的文本
        hard_intervention_prompt = hard_intervention_output.prompt  # 获取原始的输入提示
        hard_intervention_generated_text = hard_intervention_output.outputs[0].text  # 从输出对象中获取生成的文本

        formed_key_phase = get_formed_key_phase(head_entity=labeled_sample["head_entity"],
                                                tail_entity=labeled_sample["tail_entity"],
                                                given_sentence=labeled_sample["sentence"])
        original_scm_generated_text = f'Reasoning Explanations: {formed_key_phase}' + original_scm_generated_text
        hard_intervention_generated_text = f'Reasoning Explanations: {formed_key_phase}' + hard_intervention_generated_text

        original_scm = parse_original_scm_output(args=args,
                                                 output=original_scm_generated_text)
        original_scm_list.append(original_scm)

        # 对于original_scm也要进行检查，以免出现应该是无偏的，被标记为有偏的
        soft_intervention_prompt = construct_check_intervention_prompt(args=args,
                                                                       sample=labeled_sample,
                                                                       intervention_x=original_scm["x"],
                                                                       icl_samples=[selected_icl], )
        check_original_scm_prompt_list.append(soft_intervention_prompt)

        hard_intervention_x = parse_hard_intervention_output(args=args,
                                                             output=hard_intervention_generated_text)
        hard_intervention_x_list.append(hard_intervention_x)

        soft_intervention_prompt = construct_check_intervention_prompt(args=args,
                                                                       sample=labeled_sample,
                                                                       intervention_x=hard_intervention_x,
                                                                       icl_samples=[selected_icl], )
        check_hard_intervention_prompt_list.append(soft_intervention_prompt)

        print('%%%%%%%%%%%%%%%%%%%%%% original_scm prompt %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%',
              file=response_record_filepath)
        print(original_scm_prompt, file=response_record_filepath)
        print('%%%%%%%%%%%%%%%%%%%%%% original_scm generated_text %%%%%%%%%%%%%%%%%%%%%%',
              file=response_record_filepath)
        print(original_scm_generated_text, file=response_record_filepath)
        print('\n', file=response_record_filepath)

        print('%%%%%%%%%%%%%%%%%%%%%% hard_intervention prompt %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%',
              file=response_record_filepath)
        print(hard_intervention_prompt, file=response_record_filepath)
        print('%%%%%%%%%%%%%%%%%%%%%% hard_intervention generated_text %%%%%%%%%%%%%%%%%%%%%%',
              file=response_record_filepath)
        print(hard_intervention_generated_text, file=response_record_filepath)
        print('\n', file=response_record_filepath)
        print('================================================================================\n'
              '================================================================================',
              file=response_record_filepath)

    check_prompt_list = check_original_scm_prompt_list + check_hard_intervention_prompt_list

    check_outputs = llm.generate(check_prompt_list, sampling_params)
    check_original_scm_outputs = check_outputs[:len(check_original_scm_prompt_list)]
    check_hard_intervention_outputs = check_outputs[len(check_original_scm_prompt_list):]

    paired_intervention_data = []
    # 注意pair-explanation的关键信息保留，头尾实体、关系类型、原句子都要做保留

    for item_idx, (labeled_sample,
                   original_scm,
                   hard_intervention_x,
                   check_original_scm_output,
                   check_hard_intervention_output) in enumerate(zip(repeated_labeled_samples_with_x,
                                                                    original_scm_list,
                                                                    hard_intervention_x_list,
                                                                    check_original_scm_outputs,
                                                                    check_hard_intervention_outputs)):
        sample_label = labeled_sample['relation_type']

        check_original_scm_prompt = check_original_scm_output.prompt  # 获取原始的输入提示
        check_original_scm_generated_text = check_original_scm_output.outputs[0].text  # 从输出对象中获取生成的文本
        print('%%%%%%%%%%%%%%%%%%%%%% check_original_scm prompt %%%%%%%%%%%%%%%%%%%%%%', file=response_record_filepath)
        print(check_original_scm_prompt, file=response_record_filepath)
        print('%%%%%%%%%%%%%%%%%%%%%% check_original_scm generated_text %%%%%%%%%%%%%%%%%%%%%%',
              file=response_record_filepath)
        print(check_original_scm_generated_text, file=response_record_filepath)
        print('\n', file=response_record_filepath)
        print('================================================================================\n'
              '================================================================================',
              file=response_record_filepath)

        check_hard_intervention_prompt = check_hard_intervention_output.prompt  # 获取原始的输入提示
        check_hard_intervention_generated_text = check_hard_intervention_output.outputs[0].text  # 从输出对象中获取生成的文本
        print('%%%%%%%%%%%%%%%%%%%%%% check_hard_intervention prompt %%%%%%%%%%%%%%%%%%%%%%',
              file=response_record_filepath)
        print(check_hard_intervention_prompt, file=response_record_filepath)
        print('%%%%%%%%%%%%%%%%%%%%%% check_hard_intervention generated_text %%%%%%%%%%%%%%%%%%%%%%',
              file=response_record_filepath)
        print(check_hard_intervention_generated_text, file=response_record_filepath)
        print('\n', file=response_record_filepath)
        print('================================================================================\n'
              '================================================================================',
              file=response_record_filepath)

        check_original_scm_y = parse_soft_intervention_output(args=args,
                                                              output=check_original_scm_generated_text)

        soft_intervention_y = parse_soft_intervention_output(args=args,
                                                             output=check_hard_intervention_generated_text)



        if args.llm_type in ['Meta-Llama-3-70B-Instruct', 'Meta-Llama-3-70B-Instruct-GPTQ'] and  args.dataset in ['SemEval']:
            # 有个奇怪的bug，这个策略就是不输出
            check_original_scm_y = original_scm["pred_label"]
            soft_intervention_y=sample_label

        original_scm["pred_label"] = check_original_scm_y  # 注意这里是为了保证 解释和预测标签 的一致性


        if sample_label == soft_intervention_y:
            # 此时表示软干预成功->硬干预获得的结果是无偏的
            paired_intervention_data.append(
                {
                    "head_span": labeled_sample["head_entity"]["span"],
                    "tail_span": labeled_sample["tail_entity"]["span"],
                    "relation_type": sample_label,
                    "sentence_str": ' '.join(labeled_sample["sentence"]),
                    "original_scm_x": original_scm["x"],
                    "original_scm_label": original_scm["pred_label"],
                    "hard_intervention_x": hard_intervention_x,
                    "soft_intervention_y": soft_intervention_y,
                }
            )

    paired_intervention_data_filepath = \
        f'./data/{args.dataset}/sampled_{str(args.k_shot)}_shot_train_with_{args.llm_type}_paired_intervention_data.json'
    with open(paired_intervention_data_filepath, encoding='utf-8', mode='w') as f:
        # f.write(json.dumps(demonstrations_with_x))
        json.dump(paired_intervention_data, f, indent=2)
