from tqdm import trange, tqdm
from transformers import get_linear_schedule_with_warmup

from utils import GetDataLoader, filter_samples_over_length
import argparse
import json
import os

import torch
from transformers import BertTokenizer

from model import RE_BertModel

from utils import set_seeds

from utils import prepare_label_set


def test(args, model, test_samples):
    dataloader_test = GetDataLoader(args=args,
                                    samples=test_samples,
                                    batch_size=args.test_batch_size
                                    )

    metric = {"micro_f1": 0}

    num_true_micro_f1 = 0
    num_pred = 0
    num_golden = 0

    list1=[]
    list2=[]

    for batch in dataloader_test:
        _0, logits = \
            model(
                input_ids=batch[0].to(args.device),
                special_mask=batch[1].to(args.device),
                token_type_ids=batch[2].to(args.device),
                attention_mask=batch[3].to(args.device),
                labels_id=batch[4].to(args.device),
            )
        golden_labels_id = batch[4].numpy()
        pred_labels_id = torch.argmax(logits, dim=-1).cpu().numpy()

        list1.extend(golden_labels_id.tolist())
        list2.extend(pred_labels_id.tolist())

        assert len(pred_labels_id) == len(golden_labels_id)

        for idx in range(len(pred_labels_id)):

            if pred_labels_id[idx] > 0:
                # 只计算非负样本
                num_pred += 1
                if pred_labels_id[idx] == golden_labels_id[idx]:
                    num_true_micro_f1 += 1

            if golden_labels_id[idx] > 0:
                # 只计算非负样本
                num_golden += 1

    precision = num_true_micro_f1 / num_pred
    recall = num_true_micro_f1 / num_golden

    metric["micro_f1"] = 2 * (precision * recall) / (precision + recall)
    metric["precision"] = precision
    metric["recall"] = recall


    # metric["macro_f1"] = num_true_no_other / num_all_no_other
    print(metric)
    return metric


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

        for step, batch in tqdm(enumerate(dataloader_train),
                                total=len(dataloader_train),
                                desc=f"Epoch {epoch + 1}/{num_train_epochs}"):
            optimizer.zero_grad()
            # print(batch_stage1)

            loss, _1 = \
                model(
                    input_ids=batch[0].to(args.device),
                    special_mask=batch[1].to(args.device),
                    token_type_ids=batch[2].to(args.device),
                    attention_mask=batch[3].to(args.device),
                    labels_id=batch[4].to(args.device),
                )

            # print('----------- loss -----------',loss)
            # compute gradient and do step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    return model


if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='SemEval', type=str, help="dataset")

    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length")

    parser.add_argument("--train_batch_size", default=16, type=int, help="train_batch_size")
    parser.add_argument("--test_batch_size", default=16, type=int, help="test_batch_size")

    parser.add_argument("--train_learning_rate", default=2e-5, type=float, help="learning_rate")
    parser.add_argument("--train_epochs", default=100, type=int, help="train_epochs")

    #############################################################################################################
    parser.add_argument("--pretrained_model", default='bert-base-uncased', type=str, help="pretrained_model")
    parser.add_argument("--pretrained_model_hidden_size", default=768, type=int, help="hidden_size of PLM")
    #############################################################################################################

    #############################################################################################################
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--gpu_id", type=int, default=0, help="select on which gpu to train.")
    #############################################################################################################

    parser.add_argument("--k_shot", default=10, type=int, help="k_shot")

    parser.add_argument('--do_test_only', action='store_true', help='do test only')

    args = parser.parse_args()

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


    labeled_samples_filepath = f'./data/{args.dataset}/sampled_{str(args.k_shot)}_shot_train.json'
    with open(labeled_samples_filepath, encoding='utf-8') as f:
        train_samples = json.loads(f.read())

    with open(f'./data/{args.dataset}/test.json', encoding='utf-8') as f:
        test_samples = json.loads(f.read())

    ########################################################################################
    # 过滤掉过长的，避免后续代码出问题
    print('Train samples: ', len(train_samples))
    train_samples = filter_samples_over_length(args, samples=train_samples)
    print('Train samples after filtering: ', len(train_samples))

    test_samples = filter_samples_over_length(args, samples=test_samples)
    print('Test samples: ', len(test_samples))
    print('Test samples after filtering: ', len(test_samples))
    ########################################################################################

    ckpt_dir = f'./checkpoint-stage1/{args.dataset}-{str(args.seed)}-{str(args.k_shot)}-shot'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if not args.do_test_only:
        RE_model = RE_BertModel(PLM=args.pretrained_model, PLM_hidden_size=args.pretrained_model_hidden_size,
                                relation_class_num=args.relation_class_num).to(args.device)

        trained_RE_model = train(args, model=RE_model, train_samples=train_samples)
        torch.save({'model_state_dict': trained_RE_model.state_dict()},
                   os.path.join(ckpt_dir, 'bert_re_retriever.ckpt'))

    trained_RE_model = RE_BertModel(PLM=args.pretrained_model, PLM_hidden_size=args.pretrained_model_hidden_size,
                                    relation_class_num=args.relation_class_num).to(args.device)

    checkpoint_bert_model = torch.load(os.path.join(ckpt_dir, 'bert_re_retriever.ckpt'), map_location=args.device)
    trained_RE_model.load_state_dict(checkpoint_bert_model['model_state_dict'])

    all_f1 = test(args, trained_RE_model, test_samples)
