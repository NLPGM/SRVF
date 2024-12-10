import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel


def calculate_ce_loss(logits, label_ids, weight):
    ###################################
    loss_fct = CrossEntropyLoss(weight=weight)
    loss = loss_fct(logits, label_ids)
    return loss


class RE_BertModel(nn.Module):
    '''
    This is for the initial demonstration retrieval
    '''
    def __init__(self, PLM, PLM_hidden_size, relation_class_num):
        super(RE_BertModel, self).__init__()
        self.encoder = BertModel.from_pretrained(PLM)

        self.fc_layer_0 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(PLM_hidden_size, PLM_hidden_size),
        )

        self.fc_layer_1 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(PLM_hidden_size, PLM_hidden_size),
        )

        self.fc_layer_2 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(PLM_hidden_size, PLM_hidden_size),
        )

        self.linear_layer_3 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(3 * PLM_hidden_size, relation_class_num),
        )

    def forward(self, input_ids=None, special_mask=None, token_type_ids=None, attention_mask=None, labels_id=None):
        bert_outputs_raw = self.encoder(input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask
                                        )
        # bert_outputs_raw.retain_grad()
        bert_output_raw = bert_outputs_raw.last_hidden_state

        # cls_bert_output=bert_output_raw[:,0]
        # logits = self.linear_layer(cls_bert_output)

        bert_output_raw_flatten = torch.flatten(bert_output_raw, start_dim=0, end_dim=1)[:]
        special_mask_flatten = torch.flatten(special_mask, start_dim=0, end_dim=1)[:]


        # 取出special_mask为4的位置
        mask4 = special_mask_flatten == 4
        bert_output_raw_flatten4 = bert_output_raw_flatten[mask4]

        # 取出special_mask为5的位置
        mask5 = special_mask_flatten == 5
        bert_output_raw_flatten5 = bert_output_raw_flatten[mask5]

        # 取出special_mask为1的位置
        mask1 = special_mask_flatten == 1
        cls_bert_output = bert_output_raw_flatten[mask1]

        # cls_bert_output = bert_output_raw[:, 0]

        h_0 = self.fc_layer_0(cls_bert_output)
        h_1 = self.fc_layer_1(bert_output_raw_flatten4)
        h_2 = self.fc_layer_2(bert_output_raw_flatten5)
        bert_output_concat = torch.cat([h_0, h_1, h_2], dim=1)

        logits = self.linear_layer_3(bert_output_concat)

        loss = calculate_ce_loss(logits=logits,
                                 label_ids=labels_id,
                                 weight=None)

        return loss, logits

    def get_emb(self, input_ids=None, special_mask=None, token_type_ids=None, attention_mask=None):
        bert_outputs_raw = self.encoder(input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask
                                        )
        bert_output_raw = bert_outputs_raw.last_hidden_state

        bert_output_raw_flatten = torch.flatten(bert_output_raw, start_dim=0, end_dim=1)[:]
        special_mask_flatten = torch.flatten(special_mask, start_dim=0, end_dim=1)[:]

        ################################################################################
        # 取出special_mask为4的位置
        mask4 = special_mask_flatten == 4
        bert_output_raw_flatten4 = bert_output_raw_flatten[mask4]

        # 取出special_mask为5的位置
        mask5 = special_mask_flatten == 5
        bert_output_raw_flatten5 = bert_output_raw_flatten[mask5]

        # 取出special_mask为1的位置
        mask1 = special_mask_flatten == 1
        cls_bert_output = bert_output_raw_flatten[mask1]

        batch_mean_emb = (bert_output_raw_flatten4 + bert_output_raw_flatten5 + cls_bert_output) / 3
        ################################################################################



        return batch_mean_emb



class RERationaleSupervisor(nn.Module):
    '''
    This is for the iterative feedback demonstration retrieval
    '''
    def __init__(self, args, PLM, PLM_hidden_size):
        super(RERationaleSupervisor, self).__init__()
        self.args = args
        self.encoder = BertModel.from_pretrained(PLM)

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(PLM_hidden_size, PLM_hidden_size),
        )

    def forward(self, input_ids=None, special_mask=None, token_type_ids=None, attention_mask=None, labels_id=None):
        bert_outputs_raw = self.encoder(input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask
                                        )
        bert_output_raw = bert_outputs_raw.last_hidden_state

        # print(bert_output_raw.size())
        bert_output_raw_flatten = torch.flatten(bert_output_raw, start_dim=0, end_dim=1)[:]
        special_mask_flatten = torch.flatten(special_mask, start_dim=0, end_dim=1)[:]

        # 取出special_mask为1的位置
        mask1 = special_mask_flatten == 1
        cls_bert_output = bert_output_raw_flatten[mask1]

        cls_mlp_output = self.mlp(cls_bert_output)
        # print(logits.size())

        loss = self.calculate_contrastive_loss(features=cls_mlp_output,
                                               label_ids=labels_id,
                                               temperature=self.args.args_tau/100)

        return loss

    def calculate_contrastive_loss(self, features, label_ids, temperature):
        cat_labels = []
        for label_id in label_ids:
            cat_labels.append(self.args.cat_labels[label_id])
        # print(cat_labels)

        """
        calculate traditional supervised contrastive loss for comparison
        Reference: https://github.com/HobbitLong/SupContrast
        """
        diagonal = torch.eye(label_ids.shape[0], dtype=torch.bool).float().to(self.args.device)

        # 使用eq函数进行比较，得到一个二维的布尔tensor
        mask_label_equal = label_ids.unsqueeze(1).eq(label_ids.unsqueeze(0))
        # 将布尔tensor转为整型int类型的tensor
        mask_label_equal = mask_label_equal.int()

        positive_mask = mask_label_equal - diagonal  # 1 only when label is same(not include itself)

        negative_mask = 1. - mask_label_equal

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)  # 计算两两样本间点乘相似度
        # for numerical stability,减去最大正样本对的值是为了防止模型以为本行已经训练好了；
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)

        denominator = (torch.sum(exp_logits * negative_mask, dim=1, keepdim=True)
                       + torch.sum(exp_logits * positive_mask, dim=1, keepdim=True))
        log_probs = logits - torch.log(denominator)

        # 每一行的正样本对数
        num_positives_per_row = torch.sum(positive_mask, dim=1)

        # 除以每一行的正样本对数，对于没有正样本的忽略
        log_probs = (torch.sum(log_probs * positive_mask, dim=1)[num_positives_per_row > 0]
                     / num_positives_per_row[num_positives_per_row > 0])
        loss = -log_probs
        loss = loss.mean()

        return loss

    def get_emb(self, input_ids=None, special_mask=None, token_type_ids=None, attention_mask=None):
        bert_outputs_raw = self.encoder(input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask
                                        )
        bert_output_raw = bert_outputs_raw.last_hidden_state

        bert_output_raw_flatten = torch.flatten(bert_output_raw, start_dim=0, end_dim=1)[:]
        special_mask_flatten = torch.flatten(special_mask, start_dim=0, end_dim=1)[:]

        # 取出special_mask为1的位置
        mask1 = special_mask_flatten == 1
        cls_bert_output = bert_output_raw_flatten[mask1]

        return cls_bert_output


class REDiscriminator(nn.Module):
    def __init__(self, args, PLM, PLM_hidden_size):
        super(REDiscriminator, self).__init__()
        self.args = args
        self.encoder = BertModel.from_pretrained(PLM)

        self.out_linear1 = torch.nn.Linear(PLM_hidden_size, PLM_hidden_size)
        self.out_linear2 = torch.nn.Linear(PLM_hidden_size, 1)


    def forward(self,pos_input_ids, pos_special_mask, pos_token_type_ids, pos_attention_mask,
                                 neg_input_ids, neg_special_mask, neg_token_type_ids, neg_attention_mask):
        '''
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        '''
        pos_scores = self.forward_score(input_ids=pos_input_ids, special_mask=pos_special_mask, token_type_ids=pos_token_type_ids, attention_mask=pos_attention_mask)
        neg_scores = self.forward_score(input_ids=neg_input_ids, special_mask=neg_special_mask, token_type_ids=neg_token_type_ids, attention_mask=neg_attention_mask)

        ## marginrankingloss
        criterion = nn.MarginRankingLoss(margin=1, reduction='mean').to(self.args.device)
        loss = criterion(pos_scores, neg_scores, torch.ones_like(pos_scores).to(self.args.device))

        return pos_scores, neg_scores, loss

    def forward_score(self, input_ids=None, special_mask=None, token_type_ids=None, attention_mask=None):
        bert_outputs_raw = self.encoder(input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask
                                        )
        bert_output_raw = bert_outputs_raw.last_hidden_state

        bert_output_raw_flatten = torch.flatten(bert_output_raw, start_dim=0, end_dim=1)[:]
        special_mask_flatten = torch.flatten(special_mask, start_dim=0, end_dim=1)[:]

        # 取出special_mask为1的位置
        mask1 = special_mask_flatten == 1
        cls_bert_output = bert_output_raw_flatten[mask1]  # b * h

        linear_out = torch.relu(self.out_linear1(cls_bert_output))
        scores = self.out_linear2(linear_out)
        scores = torch.tanh(scores)

        return scores
