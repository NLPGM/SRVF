import argparse
import copy
import json
import random

from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from prompt_tools import parse_outputs_stage1, \
    construct_original_scm_prompt, \
    parse_reasoning_explanations_from_text, get_formed_key_phase, Output
from utils import prepare_label_set, prepare_llm, GetDataEnumerater, set_seeds, filter_samples_over_length

import os

# sys.path.append(os.path.abspath(os.path.join(__file__, "../", "..")))

from model import RE_BertModel, RERationaleSupervisor

import torch


def format_pair_intervention_data(paired_intervention_data):
    formatted_train_samples = []
    cat_labels = []

    for paired_intervention_instance in paired_intervention_data:
        head_span = paired_intervention_instance["head_span"]
        tail_span = paired_intervention_instance["tail_span"]
        relation_type = paired_intervention_instance["relation_type"]
        sentence_str = paired_intervention_instance["sentence_str"]
        original_scm_x = paired_intervention_instance["original_scm_x"]
        original_scm_label = paired_intervention_instance["original_scm_label"]
        hard_intervention_x = paired_intervention_instance["hard_intervention_x"]
        soft_intervention_y = paired_intervention_instance["soft_intervention_y"]

        filtered_original_scm_x = original_scm_x.replace(head_span, 'subject').replace(tail_span, 'object')
        cat_label = f'{original_scm_label}#{relation_type}'
        cat_labels.append(cat_label)
        formatted_sample = {
            "cat_label": cat_label,
            "rational": filtered_original_scm_x,
        }
        formatted_train_samples.append(formatted_sample)

        filtered_hard_intervention_x = hard_intervention_x.replace(head_span, 'subject').replace(tail_span, 'object')
        cat_label = f'{soft_intervention_y}#{relation_type}'  # 实际上是两个一样的relation type拼在了一起
        cat_labels.append(cat_label)
        formatted_sample = {
            "cat_label": cat_label,
            "rational": filtered_hard_intervention_x,
        }
        formatted_train_samples.append(formatted_sample)

    cat_labels = list(set(cat_labels))
    return formatted_train_samples, cat_labels


def load_stage1_retriever(args):
    # 有两种可能：simcse_retrieval or task_specific_retrieval
    if args.retrieval_stage1_mode == 'task_specific_retrieval':
        ckpt_dir = f'./checkpoint-stage1/{args.dataset}-{str(args.seed)}-{str(args.k_shot)}-shot'
        checkpoint_bert_model = torch.load(os.path.join(ckpt_dir, 'bert_re_retriever.ckpt'), map_location=args.device)
        trained_RE_model = RE_BertModel(PLM=args.retriever_plm,
                                        PLM_hidden_size=768,
                                        relation_class_num=len(args.relation_type_set)).to(args.device)
        trained_RE_model.load_state_dict(checkpoint_bert_model['model_state_dict'])
        emb_retriever_stage1 = trained_RE_model
    elif args.retrieval_stage1_mode == 'simcse_retrieval':
        # 此时不需要加载微调后的checkpoint，直接加载simcse即可
        re_bert_model = RE_BertModel(PLM=args.retriever_plm,
                                     PLM_hidden_size=768,
                                     relation_class_num=len(args.relation_type_set)).to(args.device)
        re_bert_model.encoder = BertModel.from_pretrained(args.simcse_path).to(args.device)

        emb_retriever_stage1 = re_bert_model

    return emb_retriever_stage1


def prepare_prompts_for_stage1(args, test_examples, labeled_samples_with_x):
    if 'retrieval' in args.retrieval_stage1_mode:  # 有两种可能：simcse_retrieval or task_specific_retrieval
        emb_retriever_stage1 = load_stage1_retriever(args=args)

        dataloader_test_examples = GetDataEnumerater(args=args,
                                                     samples=test_examples,
                                                     batch_size=64,
                                                     )
        dataloader_demo_examples = GetDataEnumerater(args=args,
                                                     samples=labeled_samples_with_x,
                                                     batch_size=64,
                                                     )
        logits_test_examples = None
        for out_batch_idx, batch in tqdm(enumerate(dataloader_test_examples), desc=f'Running retrieval',
                                         total=len(dataloader_test_examples)):
            batch_samples_emb = emb_retriever_stage1.get_emb(
                input_ids=batch[0].to(args.device),
                special_mask=batch[1].to(args.device),
                token_type_ids=batch[2].to(args.device),
                attention_mask=batch[3].to(args.device),
            )
            batch_logits_test_examples = None

            for inner_batch_idx, batch in enumerate(dataloader_demo_examples):
                batch_demos_emb = emb_retriever_stage1.get_emb(
                    input_ids=batch[0].to(args.device),
                    special_mask=batch[1].to(args.device),
                    token_type_ids=batch[2].to(args.device),
                    attention_mask=batch[3].to(args.device),
                )

                batch_logits = torch.matmul(batch_demos_emb, batch_samples_emb.T).detach().cpu()
                # batch_demos_emb = batch_demos_emb.detach().cpu()

                if inner_batch_idx == 0:
                    batch_logits_test_examples = batch_logits
                else:
                    batch_logits_test_examples = torch.cat((batch_logits_test_examples, batch_logits), dim=0)
                # print(batch_logits_test_examples.size())
            if out_batch_idx == 0:
                logits_test_examples = batch_logits_test_examples
            else:
                logits_test_examples = torch.cat((logits_test_examples, batch_logits_test_examples), dim=1)

        logits_test_examples = logits_test_examples.T

        # logits_test_examples 是 (示例数 * 测试样本数).T
        print(logits_test_examples.size())
        _rank_values, ranks_test_examples = logits_test_examples.topk(args.demo_num, dim=1)
        print(ranks_test_examples.size())
        ranks_test_examples = ranks_test_examples.numpy()

    elif args.retrieval_stage1_mode == 'random':
        pass

    original_scm_prompts = []
    # all_icl_ins_stage1_retrieved = []
    for test_idx, test_example in enumerate(test_examples):
        if 'retrieval' in args.retrieval_stage1_mode:
            rank_this_test = ranks_test_examples[test_idx]
            selected_demos = [labeled_samples_with_x[demo_idx] for demo_idx in rank_this_test]
        elif args.retrieval_stage1_mode == 'random':
            selected_demos = random.sample(labeled_samples_with_x, k=args.demo_num)

        # if args.dataset == 'TACRED' and args.k_shot > 50:
        #     original_scm_prompt = construct_original_scm_prompt_TACRED(args=args, sample=test_example,
        #                                                                icl_samples=selected_demos)
        # else:
        #     original_scm_prompt = construct_original_scm_prompt(args=args, sample=test_example,
        #                                                         icl_samples=selected_demos)

        original_scm_prompt = construct_original_scm_prompt(args=args, sample=test_example,
                                                            icl_samples=selected_demos)

        original_scm_prompts.append(original_scm_prompt)
        # all_icl_ins_stage1_retrieved.append(selected_demos)
    return original_scm_prompts


def load_stage2_retriever(args):
    ckpt_dir = f'./checkpoint-stage2/{args.llm_type}/{args.dataset}-{str(args.seed)}-{str(args.k_shot)}-shot'

    checkpoint_bert_model = torch.load(os.path.join(ckpt_dir, f're_contrastive_retriever-args_tau_{args.args_tau}.ckpt'),
                                           map_location=args.device)

    # the
    RE_rationale_supervisor = RERationaleSupervisor(args=args,
                                                          PLM=args.retriever_plm,
                                                          PLM_hidden_size=768).to(args.device)
    RE_rationale_supervisor.load_state_dict(checkpoint_bert_model['model_state_dict'])

    return RE_rationale_supervisor


def check_rational_after_feedback(args, rational_after_feedback_samples, paired_intervention_data,
                                  emb_retriever_stage2, CONSIDER_FALSE_SAMPLE=False):
    # 准备有偏无偏簇的原型表示；
    # 在 paired_intervention_data 中，所有的 hard_intervention_x 都属于无偏簇，归结到“unbiased_{relation_type}”原型簇中；
    # 存在两类 instance；如果original_scm_label==relation_type，则original_scm_x属于无偏簇，归结到“unbiased_{relation_type}”原型簇中；
    #                  否则，original_scm_x属于有偏簇，归结到“biased_{original_scm_label}”原型簇中；

    unbiased_clusters = {}
    biased_clusters = {}

    for paired_instance_idx, paired_intervention_instance in enumerate(paired_intervention_data):
        relation_type = paired_intervention_instance["relation_type"]
        original_scm_x = paired_intervention_instance["original_scm_x"]
        original_scm_label = paired_intervention_instance["original_scm_label"]
        hard_intervention_x = paired_intervention_instance["hard_intervention_x"]
        # soft_intervention_y = paired_intervention_instance["soft_intervention_y"]

        # 在 paired_intervention_data 中，所有的 hard_intervention_x 都属于无偏簇，归结到“unbiased_{relation_type}”原型簇中；
        if relation_type not in unbiased_clusters.keys():
            unbiased_clusters[relation_type] = [{"paired_instance_idx": paired_instance_idx,
                                                 "rational": hard_intervention_x}]
        else:
            unbiased_clusters[relation_type].append({"paired_instance_idx": paired_instance_idx,
                                                     "rational": hard_intervention_x})

        # 存在两类 instance；如果original_scm_label==relation_type，则original_scm_x属于无偏簇，归结到“unbiased_{relation_type}”原型簇中；
        if original_scm_label == relation_type:
            unbiased_clusters[relation_type].append({"paired_instance_idx": paired_instance_idx,
                                                     "rational": original_scm_x})
        else:
            # 否则，original_scm_x 属于有偏簇，归结到“biased_{original_scm_label}”原型簇中；
            if original_scm_label not in biased_clusters.keys():
                biased_clusters[original_scm_label] = [{"paired_instance_idx": paired_instance_idx,
                                                        "rational": original_scm_x}]
            else:
                biased_clusters[original_scm_label].append({"paired_instance_idx": paired_instance_idx,
                                                            "rational": original_scm_x})

    def get_rational_instances_emb(args, emb_retriever_stage2, instances):
        dataloader_instances = GetDataEnumerater(args=args,
                                                 samples=instances,
                                                 batch_size=16
                                                 )
        emb_instances = []
        for batch_idx, batch in enumerate(dataloader_instances):
            batch_mean_emb = emb_retriever_stage2.get_emb(
                input_ids=batch[0].to(args.device),
                special_mask=batch[1].to(args.device),
                token_type_ids=batch[2].to(args.device),
                attention_mask=batch[3].to(args.device),
            ).detach()
            if batch_idx == 0:
                emb_instances = batch_mean_emb
            else:
                emb_instances = torch.cat((emb_instances, batch_mean_emb), dim=0)
        return emb_instances

    collected_rational_after_feedback_samples = {}  # 收集具有同个预测标签且待纠偏的样本，集中键值为第一次预测的 test_biased_label
    status_rational_after_feedback_samples = [False] * len(
        rational_after_feedback_samples)  # 记录每一个测试样本 应不应该被 纠偏；True表示该样本有偏，应该被纠偏
    for rational_after_feedback_sample_idx, rational_after_feedback_sample in enumerate(
            rational_after_feedback_samples):
        test_biased_label = rational_after_feedback_sample["pred_label_after_feedback"]

        if test_biased_label not in collected_rational_after_feedback_samples.keys():
            collected_rational_after_feedback_samples[test_biased_label] = [rational_after_feedback_sample_idx]
        else:
            collected_rational_after_feedback_samples[test_biased_label].append(rational_after_feedback_sample_idx)
    ################################################################3

    print(unbiased_clusters.keys())

    for test_biased_label in tqdm(collected_rational_after_feedback_samples.keys(),
                                  total=len(collected_rational_after_feedback_samples.keys()),
                                  desc='Processing in collects.'):
        if not CONSIDER_FALSE_SAMPLE:
            if test_biased_label in ["Other", "no_relation"]:
                # TODO: 对于预测出来的负样本，不做处理；原因是其来源可能过于多样？
                continue

        if test_biased_label not in biased_clusters.keys():
            # 预测的标签不在有偏簇中出现过，直接不考虑
            continue

        # TODO: 在docred实验中，发现有的也不在无偏簇
        if test_biased_label not in unbiased_clusters.keys():
            # 预测的标签不在有偏簇中出现过，直接不考虑
            continue

        collected_idxes = collected_rational_after_feedback_samples[test_biased_label]
        collected_biased_rationals = [
            {"rational": rational_after_feedback_samples[rational_after_feedback_sample_idx]["rational_after_feedback"]}
            for rational_after_feedback_sample_idx in collected_idxes]

        biased_demos_cluster = biased_clusters[test_biased_label]

        biased_rationals = biased_clusters[test_biased_label]
        unbiased_rationals = unbiased_clusters[test_biased_label]

        # DONE: 根据实验情况，决定要不要加上这个  A: 不加这个；其实本来就没太多的道理
        # # 除了这个以外，需要把 biased_clusters[test_biased_label] 中涉及的无偏的 paired ins的hard_intervention_x也拿出来
        # for paired_instance_idx in [item["paired_instance_idx"] for item in biased_clusters[test_biased_label]]:
        #     unbiased_rationals.append({"rational": paired_intervention_data[paired_instance_idx]['hard_intervention_x']})

        # 这里依然先保留对于具体的样本的计算来看有无偏差，后面再考虑删去这一步看有没有副作用
        # 接着按照计算结果 把 status_test_examples 中的部分 True 改为 False （即如果与无偏的更接近，则选择不纠偏）
        # 接着，把 status_test_examples 中的其余 True 改为 具体的 pair_ins 列表（filtered_biased_pair_instances中的）

        emb_unbiased_rationals = get_rational_instances_emb(args=args,
                                                            emb_retriever_stage2=emb_retriever_stage2,
                                                            instances=unbiased_rationals)

        emb_biased_rationals = get_rational_instances_emb(args=args,
                                                          emb_retriever_stage2=emb_retriever_stage2,
                                                          instances=biased_rationals)

        emb_collected_biased_rationals = get_rational_instances_emb(args=args,
                                                                    emb_retriever_stage2=emb_retriever_stage2,
                                                                    instances=collected_biased_rationals)

        logits_unbiased_rationals = torch.matmul(emb_collected_biased_rationals, emb_unbiased_rationals.T)
        logits_biased_rationals = torch.matmul(emb_collected_biased_rationals, emb_biased_rationals.T)

        max_logit_unbiased_rationals = torch.max(logits_unbiased_rationals, dim=-1).values.tolist()
        max_logit_biased_rationals = torch.max(logits_biased_rationals, dim=-1).values.tolist()

        # print(max_logit_unbiased_rationals)
        # print(max_logit_biased_rationals)

        for local_idx, rational_after_feedback_sample_idx in enumerate(collected_idxes):
            # TODO: 根据实验情况，决定是直接相信簇计算的结果，还是再算一道（更保守） A：不能用簇，还是保留初始做法
            if max_logit_unbiased_rationals[local_idx] > max_logit_biased_rationals[local_idx]:
                status_rational_after_feedback_samples[rational_after_feedback_sample_idx] = False  # 即如果与无偏的更接近，则选择不纠偏
            else:
                # 否则，则将与 max_logit_biased_rationals 最接近的 feedback_topk 个对应的 paired_instance_idx
                # 存成列表 放在 status_test_examples[test_idx]
                # assert status_test_examples[test_idx] == True

                candidate_rank_indices = logits_biased_rationals[local_idx].topk(
                    min(args.feedback_topk * 5, len(biased_demos_cluster)),
                    dim=0).indices.cpu().numpy().tolist()

                selected_paired_instance_idxex = []
                selected_paired_instance_sentences = []

                for indice in candidate_rank_indices:
                    paired_instance_idx = biased_demos_cluster[indice]["paired_instance_idx"]
                    paired_instance_sentence = paired_intervention_data[paired_instance_idx]["sentence_str"]
                    if paired_instance_sentence not in selected_paired_instance_sentences:
                        selected_paired_instance_idxex.append(paired_instance_idx)
                        selected_paired_instance_sentences.append(paired_instance_sentence)

                    if len(selected_paired_instance_idxex) == args.feedback_topk:
                        break

                paired_ins_indices = selected_paired_instance_idxex
                paired_ins_indices.reverse()

                # 这里不能允许同一个样本示例的多次出现，
                # rank_indices = logits_biased_rationals[local_idx].topk(
                #     min(args.feedback_topk, len(biased_demos_cluster)),
                #     dim=0).indices.cpu().numpy().tolist()
                # rank_indices.reverse()
                # 该 rank_indices 指的是在 logits_biased_rationals[local_idx] 中的索引，并非 paired instance 的索引，所以要转化一下
                # paired_ins_indices = [biased_demos_cluster[rank_indice]["paired_instance_idx"] for rank_indice in
                #                       rank_indices]

                status_rational_after_feedback_samples[rational_after_feedback_sample_idx] = paired_ins_indices
    return status_rational_after_feedback_samples


def prepare_prompts_for_stage2(args, test_examples, paired_intervention_data, emb_retriever_stage2):
    # print(status_test_examples[:100])
    # 这里这么命名是为了后续循环方便
    rational_after_feedback_samples = []
    for test_idx, test_example in enumerate(test_examples):
        rational_after_feedback_samples.append(
            {
                "test_idx": test_idx,
                "pred_label_after_feedback": test_example["biased_label"],
                "rational_after_feedback": test_example["biased_explanations"],
            }
        )
    status_rational_after_feedback_samples = check_rational_after_feedback(args=args,
                                                                           rational_after_feedback_samples=rational_after_feedback_samples,
                                                                           paired_intervention_data=paired_intervention_data,
                                                                           emb_retriever_stage2=emb_retriever_stage2)

    # 因为这里 len(status_rational_after_feedback_samples)==len(rational_after_feedback_samples)
    # 所以直接进行转换
    status_test_examples = status_rational_after_feedback_samples

    # paired_intervention_data  status_test_examples  test_examples
    # 这三者可以共同，构成下一轮次的 prompt

    feedback_reconstructed_prompt_dicts = []

    for test_idx, (status, test_example) in enumerate(
            zip(status_test_examples, test_examples)):
        if status == False:
            temp_dict = {
                "prompt": '[No Feedback Here]',
            }
        else:
            feedback_icl_paired_instances = [paired_intervention_data[paired_ins_idx] for paired_ins_idx in status]
            selected_feedback_icl = []
            for paired_instance in feedback_icl_paired_instances:
                selected_feedback_icl.append(
                    {
                        "sentence": paired_instance["sentence_str"].split(' '),
                        "relation_type": paired_instance["relation_type"],
                        "explanations": paired_instance["hard_intervention_x"],
                        "head_entity": {
                            "span": paired_instance["head_span"],
                        },
                        "tail_entity": {
                            "span": paired_instance["tail_span"],
                        },
                    }
                )

            feedback_reconstructed_prompt = construct_original_scm_prompt(args=args,
                                                                          sample=test_example,
                                                                          icl_samples=selected_feedback_icl)
            temp_dict = {
                "prompt": feedback_reconstructed_prompt,
            }

        feedback_reconstructed_prompt_dicts.append(temp_dict)

    return feedback_reconstructed_prompt_dicts


def parse_outputs_stage2(args, outputs, test_examples, response_record_filepath):
    for idx, (output, test_example) in enumerate(zip(outputs, test_examples)):
        rational_after_feedback = ''

        prompt = output.prompt  # 获取原始的输入提示

        generated_text = output.outputs[0].text  # 从输出对象中获取生成的文本

        if prompt == '[No Feedback Here]':
            rational_after_feedback = '[No Feedback Here]'

            # biased_label即为第一次预测的结果，如果无偏，不需要纠正
            pred_label = test_example['biased_label']
        else:
            formed_key_phase = get_formed_key_phase(head_entity=test_example["head_entity"],
                                                    tail_entity=test_example["tail_entity"],
                                                    given_sentence=test_example["sentence"])
            generated_text = f'Reasoning Explanations: {formed_key_phase}' + generated_text

            pred_label, rational_after_feedback = parse_reasoning_explanations_from_text(args=args,
                                                                                         generated_text=generated_text)

            # if 'Prediction:' in rational_after_feedback:
            #     rational_after_feedback = rational_after_feedback.split('Prediction:')[0]

            if rational_after_feedback == '':
                rational_after_feedback = '[No Feedback Here]'

        if pred_label not in args.relation_type_set:
            pred_label = args.relation_type_set[0]  # relation_type_set的第一个就默认为 'Other'

        if rational_after_feedback != '[No Feedback Here]':
            print(f'--------样本序号：{idx}-------------', file=response_record_filepath)
            print('%%%%%%%%%%%%%%%%%%%%%% prompt %%%%%%%%%%%%%%%%%%%%%%', file=response_record_filepath)
            print(prompt, file=response_record_filepath)
            print('%%%%%%%%%%%%%%%%%%%%%% prompt %%%%%%%%%%%%%%%%%%%%%%', file=response_record_filepath)
            print(generated_text, file=response_record_filepath)
            print('########################', file=response_record_filepath)
            print(f'True label: {test_example["relation_type"]}', file=response_record_filepath)
            print('########################', file=response_record_filepath)
            print(f'初步icl更新后的预测 label: {pred_label}', file=response_record_filepath)
            print('########################', file=response_record_filepath)
            print(f'Rational after Feedback: {rational_after_feedback}', file=response_record_filepath)
            print('########################', file=response_record_filepath)
            print('--------------------------------------', file=response_record_filepath)
            print('\n', file=response_record_filepath)

        test_example['rational_after_feedback'] = rational_after_feedback
        test_example['pred_label'] = pred_label

    return test_examples


def cal_metric(test_examples_after_stage2):
    num_true = 0
    num_over_debias = 0
    num_debias_suc = 0

    num_pred = 0
    num_golden = 0
    num_true_micro_f1 = 0

    for idx, test_example in enumerate(test_examples_after_stage2):
        pred_label = test_example["pred_label"]

        if test_example["relation_type"] == pred_label:
            num_true += 1

        if pred_label not in ["Other", "no_relation"]:
            # 只计算非负样本
            num_pred += 1
        if test_example["relation_type"] not in ["Other", "no_relation"]:
            # 只计算非负样本
            num_golden += 1
        if test_example["relation_type"] == pred_label and pred_label not in ["Other", "no_relation"]:
            # 计算非负样本中预测正确的
            num_true_micro_f1 += 1

        if (test_example["biased_label"] == test_example['relation_type']
                and pred_label != test_example["relation_type"]):
            num_over_debias += 1

        if (test_example["biased_label"] != test_example['relation_type']
                and pred_label == test_example["relation_type"]):
            num_debias_suc += 1

        test_example["pred_label"] = pred_label

    acc = num_true / len(test_examples)
    rate_over_debias = num_over_debias / len(test_examples)
    rate_debias_suc = num_debias_suc / len(test_examples)

    precision = num_true_micro_f1 / num_pred
    recall = num_true_micro_f1 / num_golden
    micro_f1 = 2 * (precision * recall) / (precision + recall)

    metric = {'acc': acc,
              'micro_f1': micro_f1,
              'precision': precision,
              'recall': recall,
              'num_test_examples': len(test_examples),
              'rate_over_debias ↓': rate_over_debias,
              'rate_debias_suc  ↑': rate_debias_suc,
              'num_over_debias ↓': num_over_debias,
              'num_debias_suc  ↑': num_debias_suc,
              }
    return metric


def main_run_gpt3(args, test_examples, paired_intervention_data, labeled_samples_with_x):
    string_key_args = f'{args.dataset}_{str(args.k_shot)}_{args.retrieval_stage1_mode}_{args.llm_type}'

    metric_results_record_filepath = f'{args.prediction_results_path}/metric_results_{string_key_args}.txt'
    metric_results_record_filepath = open(metric_results_record_filepath, encoding='utf-8', mode='a')

    if args.co_server_stage == 'Server1-Stage1':
        response_record_filepath = f'{args.response_record_dir}/Server1-Stage1_{string_key_args}.txt'
    elif args.co_server_stage == 'Server2-Stage1':
        response_record_filepath = f'{args.response_record_dir}/Server2-Stage1_{string_key_args}.txt'
    elif args.co_server_stage == 'Server1-Stage2':
        response_record_filepath = f'{args.response_record_dir}/Server1-Stage2_{string_key_args}.txt'
    elif args.co_server_stage == 'Server2-Stage2':
        response_record_filepath = f'{args.response_record_dir}/Server2-Stage2_{string_key_args}.txt'
    elif args.co_server_stage == 'Server1-Stage3':
        response_record_filepath = f'{args.response_record_dir}/Server1-Stage3_{string_key_args}.txt'
    response_record_filepath = open(response_record_filepath, encoding='utf-8', mode='w')

    prompts_stage1_temp_file = f'{args.response_record_dir}/prompts_stage1_{string_key_args}.json'

    test_examples_stage1_filepath = f'{args.prediction_results_path}/test_examples_stage1_{string_key_args}.json'

    prompts_stage2_temp_file = f'{args.response_record_dir}/prompts_stage2_{string_key_args}.json'

    test_examples_stage2_API_filepath = f'{args.prediction_results_path}/test_examples_stage2_API_{string_key_args}.json'

    test_examples_stage2_filepath = f'{args.prediction_results_path}/test_examples_stage2_{string_key_args}.json'

    if args.co_server_stage == 'Server1-Stage1':
        ################################################################
        # Done by: Server 1 (w/ GPU); Server1-Stage1

        prompts_stage1 = prepare_prompts_for_stage1(args=args,
                                                    test_examples=test_examples,
                                                    labeled_samples_with_x=labeled_samples_with_x)
        json.dump(prompts_stage1, open(prompts_stage1_temp_file, encoding='utf-8', mode='w'), indent=2)
        ################################################################
    elif args.co_server_stage == 'Server2-Stage1':
        ################################################################
        # Done by: Server 2 (w/ API); Server2-Stage1
        prompts_stage1 = json.loads(open(prompts_stage1_temp_file, encoding='utf-8', mode='r').read())

        # prompts_stage1=prompts_stage1[:10]
        # test_examples=test_examples[:10]

        llm, sampling_params = prepare_llm(args=args)
        # 使用 llm.generate 方法生成输出文本。这会将输入提示加入 vLLM 引擎的等待队列，并执行引擎以高效地生成输出
        outputs_stage1 = llm.generate(prompts_stage1, sampling_params)

        test_examples_after_stage1, metric_stage1 = parse_outputs_stage1(args=args,
                                                                         outputs=outputs_stage1,
                                                                         test_examples=test_examples,
                                                                         response_record_filepath=response_record_filepath,
                                                                         )
        print(f'Report at stage1. Metric: {metric_stage1}')
        print(f'Report at stage1. Metric: {metric_stage1}', file=metric_results_record_filepath)
        json.dump(test_examples_after_stage1, open(test_examples_stage1_filepath, encoding='utf-8', mode='w'), indent=2)
        ################################################################
    elif args.co_server_stage == 'Server1-Stage2':
        ################################################################
        # Done by: Server 1 (w/ GPU); Server1-Stage2
        test_examples_after_stage1 = json.loads(open(test_examples_stage1_filepath, encoding='utf-8', mode='r').read())

        emb_retriever_stage2 = load_stage2_retriever(args=args)
        # 注意，这里的test_examples中间是含有  biased_explanations
        prompts_stage2 = prepare_prompts_for_stage2(args=args,
                                                    test_examples=test_examples_after_stage1,
                                                    paired_intervention_data=paired_intervention_data,
                                                    emb_retriever_stage2=emb_retriever_stage2)
        json.dump(prompts_stage2, open(prompts_stage2_temp_file, encoding='utf-8', mode='w'), indent=2)
        ################################################################
    elif args.co_server_stage == 'Server2-Stage2':
        ################################################################
        # Done by: Server 2 (w/ API); Server2-Stage2
        test_examples_after_stage1 = json.loads(open(test_examples_stage1_filepath, encoding='utf-8', mode='r').read())
        prompts_stage2 = json.loads(open(prompts_stage2_temp_file, encoding='utf-8', mode='r').read())
        prompts_stage2 = [item["prompt"] for item in prompts_stage2]
        llm, sampling_params = prepare_llm(args=args)
        # 使用 llm.generate 方法生成输出文本。这会将输入提示加入 vLLM 引擎的等待队列，并执行引擎以高效地生成输出
        outputs_stage2 = llm.generate(prompts_stage2, sampling_params)

        # TODO: 此变量会对传入的test_examples_after_stage1做修改，最好不要用这个
        test_examples_after_stage2 = parse_outputs_stage2(args=args,
                                                          outputs=outputs_stage2,
                                                          test_examples=test_examples_after_stage1,
                                                          response_record_filepath=response_record_filepath,
                                                          )
        # 保存为API生成的暂存文件
        json.dump(test_examples_after_stage2, open(test_examples_stage2_API_filepath, encoding='utf-8', mode='w'),
                  indent=2)
        ################################################################
    elif args.co_server_stage == 'Server1-Stage3':
        ################################################################
        # Done by: Server 1 (w/ GPU); Server1-Stage3
        emb_retriever_stage2 = load_stage2_retriever(args=args)

        test_examples_after_stage2 = json.loads(
            open(test_examples_stage2_API_filepath, encoding='utf-8', mode='r').read())
        for feedback_iter in range(1):
            rational_after_feedback_samples = []
            # 只保留 有 feedback 的结果
            for test_idx, test_example in enumerate(test_examples_after_stage2):
                rational_after_feedback = test_example["rational_after_feedback"]
                pred_label_after_feedback = test_example["pred_label"]

                if rational_after_feedback != '[No Feedback Here]':
                    rational_after_feedback_samples.append(
                        {
                            "test_idx": test_idx,
                            "pred_label_after_feedback": pred_label_after_feedback,
                            "rational_after_feedback": rational_after_feedback,
                        }
                    )

            status_rational_after_feedback_samples = check_rational_after_feedback(args=args,
                                                                                   rational_after_feedback_samples=rational_after_feedback_samples,
                                                                                   paired_intervention_data=paired_intervention_data,
                                                                                   emb_retriever_stage2=emb_retriever_stage2)
            # 如果发现纠正后的依然有偏差（即不为false），将标签置为 original_pred_label = copy.deepcopy(test_example["biased_label"])
            for tmp_idx, (rational_after_feedback_sample, status) in (
                    enumerate(zip(rational_after_feedback_samples, status_rational_after_feedback_samples))):
                test_idx = rational_after_feedback_sample["test_idx"]

                # 这里采用这样的替代方案，后续需要进一步改为重新提供反馈（可以模仿前面构造icl反馈信号的做法，但是要注意排除一些之前失败了的pair_idx）
                if status == False:
                    # 标记为'[No Feedback Here]'，在下一步时，不需要再次进行反馈纠正
                    test_examples_after_stage2[test_idx]["rational_after_feedback"] = '[No Feedback Here]'
                else:
                    test_examples_after_stage2[test_idx]["pred_label"] = (
                        test_examples_after_stage2)[test_idx]["biased_label"]

        metric = cal_metric(test_examples_after_stage2=test_examples_after_stage2)
        print(f'Report at stage2. Metric: {metric}')
        print(f'Report at stage2. Metric: {metric}', file=metric_results_record_filepath)
        print(f'{string_key_args}; \n Initial ICL demo num: {args.demo_num}; Feedback topk: {args.feedback_topk}',
              file=metric_results_record_filepath)
        print(f'-------------------------------------------------------', file=metric_results_record_filepath)
        json.dump(test_examples_after_stage2, open(test_examples_stage2_filepath, encoding='utf-8', mode='w'), indent=2)
        ################################################################


def main_run_llama(args, test_examples, paired_intervention_data, labeled_samples_with_x):
    string_key_args = f'{args.dataset}_{str(args.k_shot)}_{args.retrieval_stage1_mode}_{args.llm_type}'

    metric_results_record_filepath = f'{args.prediction_results_path}/metric_results_{string_key_args}.txt'
    metric_results_record_filepath = open(metric_results_record_filepath, encoding='utf-8', mode='a')

    if args.do_stage1:
        response_record_filepath = f'{args.response_record_dir}/stage1_{string_key_args}.txt'
    elif args.do_stage2:
        response_record_filepath = f'{args.response_record_dir}/stage2_{string_key_args}.txt'
    response_record_filepath = open(response_record_filepath, encoding='utf-8', mode='w')

    prompts_stage1_temp_file = f'{args.response_record_dir}/prompts_stage1_{string_key_args}.json'
    prompts_stage2_temp_file = f'{args.response_record_dir}/prompts_stage2_{string_key_args}.json'

    test_examples_stage1_filepath = f'{args.prediction_results_path}/test_examples_stage1_{string_key_args}.json'
    test_examples_stage2_filepath = f'{args.prediction_results_path}/test_examples_stage2_{string_key_args}.json'

    if args.do_stage1:
        # 带有初始自我解释的示例文件路径，在此之前需要进行预先准备自我解释
        # 准备带解释的标注样本集
        ######## 训练好的检索器1 加载 #########################################
        ################################################################
        prompts_stage1 = prepare_prompts_for_stage1(args=args,
                                                    test_examples=test_examples,
                                                    labeled_samples_with_x=labeled_samples_with_x)
        json.dump(prompts_stage1, open(prompts_stage1_temp_file, encoding='utf-8', mode='w'), indent=2)
        prompts_stage1 = json.loads(open(prompts_stage1_temp_file, encoding='utf-8', mode='r').read())

        ################################################################
        torch.cuda.empty_cache()
        llm, sampling_params = prepare_llm(args=args)
        # 使用 llm.generate 方法生成输出文本。这会将输入提示加入 vLLM 引擎的等待队列，并执行引擎以高效地生成输出
        outputs_stage1 = llm.generate(prompts_stage1, sampling_params)

        test_examples_after_stage1, metric_stage1 = parse_outputs_stage1(args=args,
                                                                         outputs=outputs_stage1,
                                                                         test_examples=test_examples,
                                                                         response_record_filepath=response_record_filepath,
                                                                         )
        print(f'Report at stage1. Metric: {metric_stage1}')
        print(f'Report at stage1. Metric: {metric_stage1}', file=metric_results_record_filepath)
        json.dump(test_examples_after_stage1, open(test_examples_stage1_filepath, encoding='utf-8', mode='w'), indent=2)

    elif args.do_stage2:
        # 基于阶段1的输出结果进行实验
        with open(test_examples_stage1_filepath, encoding='utf-8', mode='r') as f:
            test_examples_after_stage1 = json.loads(f.read())

        ######## 训练好的检索器2 加载 #########################################
        emb_retriever_stage2 = load_stage2_retriever(args=args)
        ################################################################

        # 1、需要维护一个 biased_label_rational = [{"test_idx":, "pred_label_after_feedback":,"rational_after_feedback":}]
        # 其记录的是当前状况下每个样本的预测结果；
        # 初始化为 {"test_idx": test_idx,"pred_label_after_feedback": test_example["biased_label"],"rational_after_feedback": test_example["biased_explanations"]}
        #         即第一阶段预测的标签和rational
        # 在每一轮的更新中，biased_label_rational 会被实时更新到最新一轮产生的预测和解释，但是，在实际输出预测结果时，并不会直接采用其结果
        #       （对于纠正后依然有偏的，会选择相信original_pred_label_rational中对应的结果）
        biased_label_rational = [{
            "test_idx": test_idx,
            "pred_label_after_feedback": test_example["biased_label"],
            "rational_after_feedback": test_example["biased_explanations"],
        } for test_idx, test_example in enumerate(test_examples_after_stage1)]

        # 2、维护一个 记录模型初始预测的 original_pred_label_rational
        # 初始化（初始化之后一直都不需要更改）类似于 biased_label_rational；
        # 此变量只读不修改；
        original_pred_label_rational = [{
            "test_idx": test_idx,
            "original_pred_label": test_example["biased_label"],
            "original_pred_rational": test_example["biased_explanations"],
        } for test_idx, test_example in enumerate(test_examples_after_stage1)]

        # 3、在以下的整个纠偏过程中，需要维护一个全局变量 status_test_samples=[]
        # 其中，每一项 为False 或相关的 建议index(对应于paired_intervention_data)；
        # 初始化：需要先检查一下第一阶段的预测与rational的状态
        status_test_examples = check_rational_after_feedback(args=args,
                                                             rational_after_feedback_samples=biased_label_rational,
                                                             paired_intervention_data=paired_intervention_data,
                                                             emb_retriever_stage2=emb_retriever_stage2)

        ################################################################
        llm, sampling_params = prepare_llm(args=args)
        ################################################################

        ori_status_test_examples = copy.deepcopy(status_test_examples)

        iters=5
        if "70" in args.llm_type:
            iters = 1
        if "13" in args.llm_type:
            iters = 2


        for feedback_iter in range(iters):

            remain_num_need_debias = 0
            # 1、并非需要对所有的样本都要进行状态检查与纠偏，只需要对 status_test_samples 中 不为 False 的进行检查

            prompts_stage2 = []  # 无论是否纠偏，都维护这样一个列表

            # 键为 test_idx 即测试样本序号，值为 在需要纠偏重新生成的所处的索引
            test_idx_to_re_generate_idx_dict = {}
            prompts_re_generate = []
            re_generate_idx = 0
            for test_idx, (status, test_example) in enumerate(zip(status_test_examples, test_examples_after_stage1)):
                if status != False:
                    # 对于 当前状态处于 需要纠偏 的测试样本
                    feedback_icl_paired_instances = [paired_intervention_data[paired_ins_idx]
                                                     for paired_ins_idx in status]
                    selected_feedback_icl = []
                    for paired_instance in feedback_icl_paired_instances:
                        selected_feedback_icl.append(
                            {
                                "sentence": paired_instance["sentence_str"].split(' '),
                                "relation_type": paired_instance["relation_type"],
                                "explanations": paired_instance["hard_intervention_x"],
                                "head_entity": {
                                    "span": paired_instance["head_span"],
                                },
                                "tail_entity": {
                                    "span": paired_instance["tail_span"],
                                },
                            }
                        )

                    feedback_reconstructed_prompt = construct_original_scm_prompt(args=args,
                                                                                  sample=test_example,
                                                                                  icl_samples=selected_feedback_icl)
                    prompts_re_generate.append(feedback_reconstructed_prompt)
                    test_idx_to_re_generate_idx_dict[test_idx] = re_generate_idx
                    re_generate_idx += 1

                    prompts_stage2.append(feedback_reconstructed_prompt)
                else:
                    prompts_stage2.append('[No Feedback Here]')

            # 重新生成
            outputs_re_generate = llm.generate(prompts_re_generate, sampling_params)

            outputs_stage2 = []
            for test_idx, prompt in enumerate(prompts_stage2):
                if test_idx in test_idx_to_re_generate_idx_dict.keys():
                    re_generate_idx = test_idx_to_re_generate_idx_dict[test_idx]
                    outputs_stage2.append(outputs_re_generate[re_generate_idx])
                else:
                    output_dict = {"prompt": prompt, "outputs": [{"text": 'None'}]}
                    # 现在我们在创建 Output 对象时传递 outputs 列表
                    output_object = Output(prompt=output_dict["prompt"], outputs=output_dict["outputs"])
                    outputs_stage2.append(output_object)

            for test_idx, (output, test_example) in enumerate(zip(outputs_stage2, test_examples_after_stage1)):

                prompt = output.prompt  # 获取原始的输入提示
                generated_text = output.outputs[0].text  # 从输出对象中获取生成的文本

                print(f'[SRVF Feedback Prompt]--------样本序号：{test_idx}-------------', file=response_record_filepath)
                print('%%%%%%%%%%%%%%%%%%%%%% prompt %%%%%%%%%%%%%%%%%%%%%%', file=response_record_filepath)
                print(prompt, file=response_record_filepath)
                print('%%%%%%%%%%%%%%%%%%%%%% generated_text %%%%%%%%%%%%%%%%%%%%%%', file=response_record_filepath)
                print(generated_text, file=response_record_filepath)
                print('########################', file=response_record_filepath)
                print('--------------------------------------', file=response_record_filepath)
                print('\n', file=response_record_filepath)


                if status_test_examples[test_idx] != False:
                    # 只有在此时，即当前处于 被纠正状态 的样本，才需要进行下面的解析
                    formed_key_phase = get_formed_key_phase(head_entity=test_example["head_entity"],
                                                            tail_entity=test_example["tail_entity"],
                                                            given_sentence=test_example["sentence"])
                    generated_text = f'Reasoning Explanations: {formed_key_phase}' + generated_text

                    pred_label, pred_rational = parse_reasoning_explanations_from_text(args=args,
                                                                                       generated_text=generated_text)

                    if pred_label not in args.relation_type_set:
                        pred_label = args.relation_type_set[0]  # relation_type_set的第一个就默认为 'Other'

                    # 此时需要对 biased_label_rational 进行更新；
                    assert biased_label_rational[test_idx]["test_idx"] == test_idx

                    biased_label_rational[test_idx]["pred_label_after_feedback"] = pred_label
                    biased_label_rational[test_idx]["rational_after_feedback"] = pred_rational

            # 注意，这里所要检查的 偏差预测并不是和初始化时的一样（对所有的都检查）
            # 而是只检查当前 status 不为 False 的；且在赋予被检查 rational是也是 biased_label_rational 这一全局变量所存储的

            test_idx_to_remain_idx_dict = {}
            remain_idx = 0

            remain_biased_label_rational = []
            for test_idx, (status, test_example) in enumerate(zip(status_test_examples, test_examples_after_stage1)):
                if status != False:
                    # 表示在本轮进行了重新生成的
                    remain_biased_label_rational.append({
                        "test_idx": test_idx,
                        "pred_label_after_feedback": biased_label_rational[test_idx]["pred_label_after_feedback"],
                        "rational_after_feedback": biased_label_rational[test_idx]["rational_after_feedback"], }
                    )

                    test_idx_to_remain_idx_dict[test_idx] = remain_idx
                    remain_idx += 1

            remain_status_test_examples = check_rational_after_feedback(args=args,
                                                                        rational_after_feedback_samples=remain_biased_label_rational,
                                                                        paired_intervention_data=paired_intervention_data,
                                                                        emb_retriever_stage2=emb_retriever_stage2, )
            # 依据上面的状态，对于 status_test_examples 进行更新，如果依然不是False的，则更新为新一轮的建议的[pair_idx,...]
            for test_idx, status in enumerate(status_test_examples):
                if test_idx in test_idx_to_remain_idx_dict.keys():
                    remain_idx = test_idx_to_remain_idx_dict[test_idx]
                    remain_status = remain_status_test_examples[remain_idx]
                    if remain_status == False:
                        # 此时更新 status_test_examples 中相关项为False；即此时可以信任 biased_label_rational 中的当前值
                        status_test_examples[test_idx] = False
                    else:

                        # 说明依然存在偏差，则需要更新为最新的建议（remain_status为 建议的[pair_idx,...]）
                        status_test_examples[test_idx] = remain_status
                        remain_num_need_debias += 1

            # 完成了上述的：修正、检查、重新赋状态值；可以对test_examples中具体样本的 ["pred_label"] 做赋值
            for test_idx, (status, test_example) in enumerate(zip(status_test_examples, test_examples_after_stage1)):
                if status == False:
                    # 此时可以信任 biased_label_rational 中的当前值
                    # （一种是此轮纠偏前就已经是False的，则直接信任最初结果）
                    # （另一种是，此轮纠偏前不是False的，但是纠偏后更新了 biased_label_rational 中的相关值，同时检查发现此次纠偏结果可信任）
                    test_example["pred_label"] = biased_label_rational[test_idx]["pred_label_after_feedback"]
                    test_example["pred_rational"] = biased_label_rational[test_idx]["rational_after_feedback"]

                else:
                    # 此时不可信任 biased_label_rational 中的当前值；采取 original_pred_label_rational 中的相关值；
                    # （这里只是为了保留一个预测值，在实际情况中，随着多轮纠偏的发生，上面的if会被经进入
                    test_example["pred_label"] = original_pred_label_rational[test_idx]["original_pred_label"]
                    test_example["pred_rational"] = original_pred_label_rational[test_idx]["original_pred_rational"]

            # 后续的循环将会再次 基于 status_test_examples 判断每个样本的状态（要不要相信 biased_label_rational ），
            # 并对应地调整 biased_label_rational；
            # 并多次调整 test_examples_after_stage1 中的值
            # 下面是对 test_examples_after_stage1 中相关 指标值的计算

            num_true = 0
            num_over_debias = 0
            num_debias_suc = 0

            num_pred = 0
            num_golden = 0
            num_true_micro_f1 = 0

            for test_idx, test_example in enumerate(test_examples_after_stage1):
                pred_label = test_example["pred_label"]
                golden_label = test_example["relation_type"]

                if test_example["relation_type"] == pred_label:
                    num_true += 1

                if pred_label not in ["Other", "no_relation"]:
                    # 只计算非负样本
                    num_pred += 1
                    if test_example["relation_type"] == pred_label:
                        num_true_micro_f1 += 1
                if test_example["relation_type"] not in ["Other", "no_relation"]:
                    # 只计算非负样本
                    num_golden += 1

                original_pred_label = original_pred_label_rational[test_idx]["original_pred_label"]

                if original_pred_label == golden_label and pred_label != golden_label:
                    # 如果原始预测是对的，但是当前预测是错的，说明是过度纠偏
                    num_over_debias += 1

                if original_pred_label != golden_label and pred_label == golden_label:
                    # 如果原始预测是错的，但是当前预测是对的，说明是成功纠偏
                    num_debias_suc += 1

            acc = num_true / len(test_examples_after_stage1)
            rate_over_debias = num_over_debias / len(test_examples_after_stage1)
            rate_debias_suc = num_debias_suc / len(test_examples_after_stage1)

            precision = num_true_micro_f1 / num_pred
            recall = num_true_micro_f1 / num_golden
            micro_f1 = 2 * (precision * recall) / (precision + recall)

            metric = {'acc': acc,
                      'micro_f1': micro_f1,
                      'precision': precision,
                      'recall': recall,
                      'num_test_examples': len(test_examples_after_stage1),
                      'rate_over_debias ↓': rate_over_debias,
                      'rate_debias_suc  ↑': rate_debias_suc,
                      'num_over_debias ↓': num_over_debias,
                      'num_debias_suc  ↑': num_debias_suc,
                      }
            print(f'Report at stage2. 反馈轮次：{feedback_iter + 1}\n Metric: {metric}')
            print(f'remain_num_need_debias: {remain_num_need_debias}')

            print(f'Report at stage2. 反馈轮次：{feedback_iter + 1}\n Metric: {metric}',
                  file=metric_results_record_filepath)
            print(f'remain_num_need_debias: {remain_num_need_debias}', file=metric_results_record_filepath)

        test_examples_after_stage2 = test_examples_after_stage1
        print(f'Report at stage2. Final Metric: {metric}')
        print(f'Report at stage2. Final Metric: {metric}', file=metric_results_record_filepath)
        print(f'{string_key_args}; \n Initial ICL demo num: {args.demo_num}; Feedback topk: {args.feedback_topk}',
              file=metric_results_record_filepath)
        print(f'***************** args_tau: {args.args_tau}')
        print(f'***************** args_tau: {args.args_tau}', file=metric_results_record_filepath)
        print(f'-------------------------------------------------------', file=metric_results_record_filepath)
        print(f'-------------------------------------------------------', file=metric_results_record_filepath)
        json.dump(test_examples_after_stage2, open(test_examples_stage2_filepath, encoding='utf-8', mode='w'), indent=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--seed', type=int, default=42, help='Set a random seed')
    parser.add_argument('--dataset', type=str, default='SemEval', help='Choose one dataset for experiments.')
    # parser.add_argument('--num_test_examples', type=int, default=1000, help='Set the number of test examples')
    parser.add_argument('--k_shot', type=int, default=10, help='number of demo examples per class')
    parser.add_argument('--llm_type', type=str, default='llama2_7b_chat', help='llm type ')

    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='max seq length, used for retriever truncation.')

    parser.add_argument('--demo_num', type=int, default=10, help='number of demonstration examples')

    parser.add_argument('--feedback_topk', type=int, default=6, help='number of demonstration examples')

    parser.add_argument('--retrieval_stage1_mode', type=str,
                        default='retrieval',
                        help='in [random, simcse_retrieval, task_specific_retrieval]')

    parser.add_argument('--retriever_plm', type=str, default='bert-base-uncased',
                        help='plm used for retrievers')

    parser.add_argument('--simcse_path', type=str, default='princeton-nlp/sup-simcse-bert-base-uncased',
                        help='the path for simcse pretrained LM')


    parser.add_argument('--do_stage1', action='store_true', help='内存限制，stage1和2分开做')
    parser.add_argument('--do_stage2', action='store_true', help='内存限制，stage1和2分开做')

    parser.add_argument('--co_server_stage',
                        type=str,
                        default='Server1-Stage1',
                        help="由于服务器访问API受限制，所以在gpt-3.5下需要用这种多协作方式 "
                             "['Server1-Stage1', 'Server1-Stage2', 'Server1-Stage3', 'Server2-Stage1', 'Server2-Stage2']")

    parser.add_argument('--do_local', action='store_true', help='网络限制，本地运行')

    parser.add_argument('--prediction_results_path', type=str, default='prediction_results', help='prediction results')

    parser.add_argument('--args_tau', type=int, default=20, help='20/100=0.2 as the tau parameter')

    args = parser.parse_args()
    set_seeds(args)

    ################################################################
    # 1、自动检测是否支持CUDA，并据此选择使用GPU还是CPU
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cuda:" + str(1) if torch.cuda.is_available() else "cpu")

    print(f"Using device: {args.device}")

    ################################################################
    # 2、设置本实验中所用retriever所用到的tokenizer
    setattr(args, 'tokenizer', BertTokenizer.from_pretrained(args.retriever_plm))

    ################################################################
    # 3、数据读取与初步处理
    labeled_samples_with_x_filepath = f'./data/{args.dataset}/sampled_{str(args.k_shot)}_shot_train_with_{args.llm_type}_x.json'

    labeled_samples_with_x = json.loads(open(labeled_samples_with_x_filepath, mode='r', encoding='utf-8').read())

    #########################################################################################################
    # 对于全样本 TACRED 阶段1随机选取示例的设置，需要人为地设置候选随机示例的范围

    #########################################################################################################

    test_filepath = f'./data/{args.dataset}/test.json'
    test_examples = json.load(open(test_filepath, 'r', encoding='utf-8'))
    # test_examples = test_examples[: args.num_test_examples]

    # 过滤掉过长的，避免后续代码出问题
    print('Labeled samples: ', len(labeled_samples_with_x))
    labeled_samples_with_x = filter_samples_over_length(args, samples=labeled_samples_with_x)
    print('Labeled samples after filtering: ', len(labeled_samples_with_x))

    test_examples = filter_samples_over_length(args, samples=test_examples)
    print('Test samples: ', len(test_examples))
    print('Test samples after filtering: ', len(test_examples))

    ################################################################
    paired_intervention_data_filepath = f'./data/{args.dataset}/sampled_{str(args.k_shot)}_shot_train_with_{args.llm_type}_paired_intervention_data.json'

    with open(paired_intervention_data_filepath, encoding='utf-8', mode='r') as f:
        paired_intervention_data = json.loads(f.read())
    _, cat_labels = format_pair_intervention_data(paired_intervention_data)
    setattr(args, 'cat_labels_num', len(cat_labels))

    ########################################################################################
    # 4、准备类别标签集合
    relation_type_set, relation_set_dict = prepare_label_set(dataset=args.dataset)
    setattr(args, 'relation_type_set', relation_type_set)
    setattr(args, 'relation_set_dict', relation_set_dict)

    ################################################################
    # 5、准备prompt记录文件夹
    args.response_record_dir = './response_record_test_examples'
    if args.llm_type == 'gpt-3.5-turbo':
        args.response_record_dir=args.response_record_dir.replace("response_record_test_examples","API_response_record")

    if not os.path.exists(args.response_record_dir):
        os.mkdir(args.response_record_dir)

    args.prediction_results_path = './prediction_results'
    if args.llm_type == 'gpt-3.5-turbo':
        args.prediction_results_path=args.prediction_results_path.replace("prediction_results","API_prediction_results")

    if not os.path.exists(args.prediction_results_path):
        os.mkdir(args.prediction_results_path)

    ################################################################
    # 6、固定初始ICL数目
    if args.dataset in ['TACRED', 'Re-TACRED', 'DOCRED']:
        args.demo_num = 4
    elif args.dataset in ['SemEval']:
        args.demo_num = 10

    if args.llm_type == 'gpt-3.5-turbo':
        # test_examples = test_examples[: 1000]
        main_run_gpt3(args, test_examples, paired_intervention_data, labeled_samples_with_x)
    else:
        main_run_llama(args, test_examples, paired_intervention_data, labeled_samples_with_x)
