from tqdm import trange
from transformers import get_linear_schedule_with_warmup

from utils import GetDataLoader, prepare_label_set
import argparse
import json
import os

import torch
from transformers import BertTokenizer

from model import RERationaleSupervisor

from utils import set_seeds



def train(args, model, train_samples):
    dataloader_train = GetDataLoader(args=args,
                                     samples=train_samples,
                                     batch_size=args.train_batch_size
                                     )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.train_learning_rate)

    num_train_epochs = args.train_epochs
    num_update_steps_per_epoch = len(dataloader_train)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # epoch_iterator = trange(0, args.train_epochs, desc="train epochs", disable=False)
    model.train()

    for epoch in trange(args.train_epochs):

        # batch_iterator = tqdm(dataloader_train, desc="batch iterator", disable=False)
        batch_loss=0
        for step, batch in enumerate(dataloader_train):
            optimizer.zero_grad()
            # print(batch_stage1)

            loss = \
                model(
                    input_ids=batch[0].to(args.device),
                    special_mask=batch[1].to(args.device),
                    token_type_ids=batch[2].to(args.device),
                    attention_mask=batch[3].to(args.device),
                    labels_id=batch[4].to(args.device),
                )
            # 对于某些为nan，即无正样本对的
            if torch.isnan(loss).any():
                continue

            # compute gradient and do step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            batch_loss+=loss
        print('----------- epoch loss -----------',batch_loss)

    return model


def format_pair_intervention_data(paired_intervention_data):
    formatted_train_samples = []
    cat_labels = []

    processed_sentence_strs=[]

    for paired_intervention_instance in paired_intervention_data:
        head_span = paired_intervention_instance["head_span"]
        tail_span = paired_intervention_instance["tail_span"]
        relation_type = paired_intervention_instance["relation_type"]
        sentence_str = paired_intervention_instance["sentence_str"]
        original_scm_x = paired_intervention_instance["original_scm_x"]
        original_scm_label = paired_intervention_instance["original_scm_label"]
        hard_intervention_x = paired_intervention_instance["hard_intervention_x"]
        soft_intervention_y = paired_intervention_instance["soft_intervention_y"]

        cat_label = f'{original_scm_label}#{relation_type}'
        cat_labels.append(cat_label)
        formatted_sample = {
            "cat_label": cat_label,
            "rational": original_scm_x,
        }
        formatted_train_samples.append(formatted_sample)

        if sentence_str not in processed_sentence_strs:

            cat_label = f'{soft_intervention_y}#{relation_type}' # 实际上是两个一样的relation type拼在了一起
            cat_labels.append(cat_label)
            formatted_sample = {
                "cat_label": cat_label,
                "rational": hard_intervention_x,
            }
            formatted_train_samples.append(formatted_sample)

        processed_sentence_strs.append(sentence_str)

    cat_labels = list(set(cat_labels))
    return formatted_train_samples, cat_labels


if __name__ == '__main__':

    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='SemEval', type=str, help="dataset")

    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length")

    parser.add_argument("--train_batch_size", default=128, type=int, help="train_batch_size")
    parser.add_argument("--test_batch_size", default=16, type=int, help="test_batch_size")

    parser.add_argument("--train_learning_rate", default=2e-5, type=float, help="learning_rate")
    parser.add_argument("--train_epochs", default=50, type=int, help="train_epochs")

    #############################################################################################################
    parser.add_argument("--pretrained_model", default='bert-base-uncased', type=str, help="pretrained_model")
    parser.add_argument("--pretrained_model_hidden_size", default=768, type=int, help="hidden_size of PLM")
    #############################################################################################################

    #############################################################################################################
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--gpu_id", type=int, default=0, help="select on which gpu to train.")
    #############################################################################################################

    parser.add_argument("--k_shot", default=10, type=int, help="k_shot")
    parser.add_argument('--llm_type', type=str, default='llama2_7b_chat', help='llm type ')
    parser.add_argument('--args_tau', type=int, default=20, help='20/100=0.2 as the tau parameter')

    args = parser.parse_args()

    llm_type = args.llm_type



    ######################################
    relation_type_set, relation_set_dict = prepare_label_set(dataset=args.dataset)
    setattr(args, 'relation_type_set', relation_type_set)
    setattr(args, 'relation_set_dict', relation_set_dict)
    setattr(args, 'relation_class_num', len(relation_type_set))

    print('***************** working on gpu id: ', args.gpu_id, ' *****************')
    args.device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")

    setattr(args, 'tokenizer', BertTokenizer.from_pretrained(args.pretrained_model))

    args.n_gpu = 0 if not torch.cuda.is_available else torch.cuda.device_count()
    args.n_gpu = min(1, args.n_gpu)
    set_seeds(args)


    paired_intervention_data_filepath = \
        f'./data/{args.dataset}/sampled_{str(args.k_shot)}_shot_train_with_{args.llm_type}_paired_intervention_data.json'
    with open(paired_intervention_data_filepath, encoding='utf-8', mode='r') as f:
        paired_intervention_data = json.loads(f.read())

    formatted_train_samples, cat_labels = format_pair_intervention_data(paired_intervention_data)

    cat_labels_dict = {}
    for idx, item in enumerate(cat_labels):
        cat_labels_dict[item] = idx
    setattr(args, 'cat_labels', cat_labels)
    setattr(args, 'cat_labels_dict', cat_labels_dict)
    setattr(args, 'cat_labels_num', len(cat_labels))

    ckpt_dir = f'./checkpoint-stage2/{args.llm_type}/{args.dataset}-{str(args.seed)}-{str(args.k_shot)}-shot'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    RE_contrastive_retriever = RERationaleSupervisor(args=args,
                                                      PLM=args.pretrained_model,
                                                      PLM_hidden_size=args.pretrained_model_hidden_size).to(args.device)
    trained_RE_contrastive_retriever = train(args,
                                             model=RE_contrastive_retriever,
                                             train_samples=formatted_train_samples)

    torch.save({'model_state_dict': trained_RE_contrastive_retriever.state_dict()},
               os.path.join(ckpt_dir, f're_contrastive_retriever-args_tau_{args.args_tau}.ckpt'))
