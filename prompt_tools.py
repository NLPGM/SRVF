import re
import time

from openai import OpenAI
from tqdm import tqdm


def get_prompt_for_initial_rational(args, labeled_sample):
    if args.dataset in ['TACRED','Re-TACRED']:
        basic_RE_X_prompt = '''Instruction: Given a sentence, explain why there is certain relation between the head and tail entities in the sentence.
        Demonstration:
        <Start of Instance>
        Given Sentence: "Space X was founded by Musk."
        Head Entity: "Space X"
        Tail Entity: "Musk"
        The relation type between "Space X" and "Musk" is "org:founded_by"
        Reasoning Explanations: In the given sentence, the key phrase "was founded by" implies that the company "Space X" was created by the person "Musk". Therefore, the head entity "Space X" serves as the "org" while the tail entity "Musk" servers as the "founded_by" person. 
        Given the sentence, the relation between the head entity "Space X" and the tail entity "Musk" is "org:founded_by"
        <End of Instance>
        <Hint> 
        Please learn the demonstration and follow the instruction, output the explanations part of the new given instance. 
        Please end with <End of Instance> when complete the text.
        <Hint>
        <Start of Instance>
        Given Sentence: "{given_sentence}"
        Head Entity: "{head_entity}"
        Tail Entity: "{tail_entity}"
        The relation type between "{head_entity}" and "{tail_entity}" is "{relation_type}"
        Reasoning Explanations: {formed_key_phase}'''
    elif args.dataset in ['SemEval']:
        basic_RE_X_prompt = '''Instruction: Given a sentence, explain why there is certain relation between the head and tail entities in the sentence.
        Demonstration:
        <Start of Instance>
        Given Sentence: "The therapist treats the patient with a certain kind of manual therapy."
        Head Entity: "therapist"
        Tail Entity: "therapy"
        The relation type between "therapist" and "therapy" is "Instrument-Agency"
        Reasoning Explanations: In the given sentence, the key phrase "treats the patient with" implies that the therapy is the tool employed by the therapist to treat the patient. Therefore, the head entity "therapy" serves as the "Instrument" while the tail entity "therapist" servers as the "Agency". 
        Given the sentence, the relation between the head entity "therapy" and the tail entity "therapist" is "Instrument-Agency"
        <End of Instance>
        <Hint> 
        Please learn the demonstration and follow the instruction, output the explanations part of the new given instance. 
        Please end with <End of Instance> when complete the text.
        <Hint>
        <Start of Instance>
        Given Sentence: "{given_sentence}"
        Head Entity: "{head_entity}"
        Tail Entity: "{tail_entity}"
        The relation type between "{head_entity}" and "{tail_entity}" is "{relation_type}"
        Reasoning Explanations: {formed_key_phase}'''
    elif args.dataset in ['DOCRED']:
        basic_RE_X_prompt = '''Instruction: Given a sentence, explain why there is certain relation between the head and tail entities in the sentence.
        Demonstration:
        <Start of Instance>
        Given Sentence: "She testified at the US Congress, which is located in New York in United Nations."
        Head Entity: "US Congress"
        Tail Entity: "United Nations"
        The relation type between "US Congress" and "United Nations" is "headquarters location"
        Reasoning Explanations: In the given sentence, the key phrase "which is located in New York in" implies that the "United Nations" is the "headquarters location" of "US Congress". Therefore, the relation between the head entity "US Congress" and the tail entity "United Nations" is "headquarters location".
        Given the sentence, the relation between the head entity "US Congress" and the tail entity "United Nations" is "headquarters location".
        <End of Instance>
        <Hint> 
        Please learn the demonstration and follow the instruction, output the explanations part of the new given instance. 
        Please end with <End of Instance> when complete the text.
        <Hint>
        <Start of Instance>
        Given Sentence: "{given_sentence}"
        Head Entity: "{head_entity}"
        Tail Entity: "{tail_entity}"
        The relation type between "{head_entity}" and "{tail_entity}" is "{relation_type}"
        Reasoning Explanations: {formed_key_phase}'''


    basic_RE_X_prompt = basic_RE_X_prompt.replace('\n        ', '\n')

    test_given_sentence = labeled_sample["sentence"]
    test_head_entity = labeled_sample["head_entity"]
    test_tail_entity = labeled_sample["tail_entity"]
    relation_type = labeled_sample["relation_type"]

    formed_key_phase = get_formed_key_phase(head_entity=test_head_entity, tail_entity=test_tail_entity,
                                            given_sentence=test_given_sentence)

    re_x_prompt = basic_RE_X_prompt.format(
        given_sentence=' '.join(test_given_sentence),
        head_entity=test_head_entity["span"],
        tail_entity=test_tail_entity["span"],
        relation_type=relation_type,
        formed_key_phase=formed_key_phase,
    )
    return re_x_prompt

def parse_reasoning_explanations_from_text(args,generated_text):
    if '<End of Instance>' in generated_text:
        generated_text = generated_text.split('<End of Instance>')[0]

    if 'Reasoning Explanations:' in generated_text:
        reasoning_explanations = generated_text.split('Reasoning Explanations:')[1]
    else:
        if 'Reasoning Explanation:' in generated_text:
            reasoning_explanations = generated_text.split('Reasoning Explanation:')[1]

        else:
            reasoning_explanations = ''

    reasoning_explanations = reasoning_explanations.split('Prediction')[0].split('Correct')[0]
    reasoning_explanations = reasoning_explanations.replace('\n', '')

    if 'Prediction' in generated_text:
        text_for_match = generated_text.split('Prediction')[1]
    else:
        text_for_match = generated_text

    # Re-TACRED中有bug，出现bug时解析有问题；例如 'per:cities_of_residence.' 识别不出来（只在gpt-3.5中发现该现象）
    text_for_match=text_for_match.replace('.','')

    matches=[]
    if args.dataset in ['TACRED','Re-TACRED']:
        matches = re.findall(r'"[^"]+:[^"]+"', text_for_match)
    elif args.dataset in ['SemEval']:
        matches = re.findall(r'"[^"]+-[^"]+"', text_for_match)
    elif args.dataset in ['DOCRED']:
        matches = re.findall(r'is "(.*?)"', text_for_match)
    # print(matches)
    if len(matches) > 0:
        if args.dataset in ['TACRED','Re-TACRED','SemEval']:
            pred_label = matches[0]
            pred_label = pred_label[1:-1]
        elif args.dataset in ['DOCRED']:
            pred_label = matches[0]

    else:
        pred_label = 'None'  # 这个none在具体的调用后会转成other或无标签

    if pred_label not in args.relation_type_set:
        pred_label = args.relation_type_set[0]  # relation_type_set的第一个就默认为 'Other'

    return pred_label, reasoning_explanations





def get_formed_key_phase(head_entity, tail_entity, given_sentence):
    real_start = min(head_entity["start_idx"], tail_entity["start_idx"])
    real_end = max(head_entity["end_idx"], tail_entity["end_idx"])
    key_phase = ' '.join(given_sentence[real_start:real_end + 1])
    formed_key_phase = f'In the given sentence, the key phrase "{key_phase}" implies that'
    return formed_key_phase


def parse_outputs_stage1(args, outputs, test_examples, response_record_filepath):
    num_true = 0

    num_true_micro_f1 = 0
    num_pred=0
    num_golden=0


    for idx, (output, test_example) in enumerate(
            zip(outputs, test_examples)):
        prompt = output.prompt  # 获取原始的输入提示
        generated_text = output.outputs[0].text  # 从输出对象中获取生成的文本

        formed_key_phase = get_formed_key_phase(head_entity=test_example["head_entity"],
                                                tail_entity=test_example["tail_entity"],
                                                given_sentence=test_example["sentence"])
        generated_text = f'Reasoning Explanations: {formed_key_phase}' + generated_text

        pred_label, bias_explanations = parse_reasoning_explanations_from_text(args=args,generated_text=generated_text)



        if test_example["relation_type"] != pred_label:
            print(f'--------样本序号：{idx}-------------', file=response_record_filepath)
            print('%%%%%%%%%%%%%%%%%%%%%% prompt %%%%%%%%%%%%%%%%%%%%%%', file=response_record_filepath)
            print(prompt, file=response_record_filepath)
            print('%%%%%%%%%%%%%%%%%%%%%% prompt %%%%%%%%%%%%%%%%%%%%%%', file=response_record_filepath)
            print(generated_text, file=response_record_filepath)
            print('########################', file=response_record_filepath)
            print(test_example["relation_type"], file=response_record_filepath)
            print('########################', file=response_record_filepath)

            print('########################', file=response_record_filepath)
            print(f'Bias pred label: {pred_label}', file=response_record_filepath)
            print(f'Bias explanations: {bias_explanations}', file=response_record_filepath)
            print('########################', file=response_record_filepath)
            print('--------------------------------------', file=response_record_filepath)
            print('\n', file=response_record_filepath)

        if test_example["relation_type"] == pred_label:
            num_true += 1

        if pred_label not in ["Other","no_relation"]:
            # 只计算非负样本
            num_pred += 1
        if test_example["relation_type"] not in ["Other","no_relation"]:
            # 只计算非负样本
            num_golden += 1
        if test_example["relation_type"] == pred_label and pred_label not in ["Other","no_relation"]:
            # 计算非负样本中预测正确的
            num_true_micro_f1 += 1

        test_example['biased_explanations'] = bias_explanations
        test_example['biased_label'] = pred_label

    acc = num_true / len(test_examples)

    precision = num_true_micro_f1 / num_pred
    recall = num_true_micro_f1 / num_golden
    micro_f1 = 2 * (precision * recall) / (precision + recall)

    metric = {'acc': acc,
              'micro_f1':micro_f1,
              'precision': precision,
              'recall': recall,
              'num_test_examples': len(test_examples)}
    print(f'metric of stage1: {metric}', file=response_record_filepath)

    return test_examples, metric


class ChatGPT_Class():
    def __init__(self, llm_type):
        super(ChatGPT_Class, self).__init__()
        self.key = ''
        self.client = OpenAI(api_key=self.key)
        self.llm_type = llm_type

    def generate(self, prompts, sampling_params):
        outputs = []
        for prompt in tqdm(prompts[:], desc='Generating using GPT-3.5'):
            if prompt in ["[No Feedback Here]", ]:
                output_dict = {
                    "prompt": prompt,
                    "outputs": [
                        {
                            "text": 'None'
                        }
                    ]
                }
                # 现在我们在创建 Output 对象时传递 outputs 列表
                output_object = Output(prompt=output_dict["prompt"], outputs=output_dict["outputs"])
                outputs.append(output_object)
                continue
            try:
                completion = self.client.chat.completions.create(
                    model=self.llm_type,
                    messages=[
                        {"role": "system", "content": "You are a text completion assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1024,
                    temperature=sampling_params['temperature'],
                    n=sampling_params['n'],
                )
                generated_text = completion.choices[0].message.content
            except:
                time.sleep(100)
                completion = self.client.chat.completions.create(
                    model=self.llm_type,
                    messages=[
                        {"role": "system", "content": "You are a text completion assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1024,
                    temperature=sampling_params['temperature'],
                    n=sampling_params['n'],
                )
                generated_text = completion.choices[0].message.content

            # print(prompt)
            # print(generated_text)

            output_dict = {
                "prompt": prompt,
                "outputs": [
                    {
                        "text": generated_text
                    }
                ]
            }

            # 现在我们在创建 Output 对象时传递 outputs 列表
            output_object = Output(prompt=output_dict["prompt"], outputs=output_dict["outputs"])

            outputs.append(output_object)
        return outputs


class OutputText:
    def __init__(self, text):
        self.text = text


class Output:
    def __init__(self, prompt, outputs):
        self.prompt = prompt
        self.outputs = [OutputText(text=output["text"]) for output in outputs]


def construct_original_scm_prompt(args, sample, icl_samples):
    basic_instruction_prompt = '''Instruction: Determine the relation between the given head entity and tail entity in the given sentence. The relation category is from the relation type set.
    Demonstrations:
    {demonstrations}
    <Hint> 
    Please learn the demonstration and follow the instruction, complete the "Reasoning Explanations" and "Prediction" parts of the new given instance. 
    You only need to solve the only instance given. 
    Please end with <End of Instance> when complete the text.
    <Hint>
    Inference:
    <Start of Instance>
    Given Sentence: "{given_sentence}"
    Relation Type Set: {relation_type_set}
    Head Entity: "{head_entity}"
    Tail Entity: "{tail_entity}"
    Reasoning Explanations: {formed_key_phase}'''
    basic_instruction_prompt = basic_instruction_prompt.replace('\n    ', '\n')

    basic_icl_prompt = '''<Start of Instance>
    Given Sentence: "{given_sentence}"
    Relation Type Set: {relation_type_set}
    Head Entity: "{head_entity}"
    Tail Entity: "{tail_entity}"
    Reasoning Explanations: {explanations}
    Prediction: Given the sentence, the relation between the head entity "{head_entity}" and the tail entity "{tail_entity}" is "{relation_type}"
    <End of Instance>
    '''

    basic_icl_prompt = basic_icl_prompt.replace('\n    ', '\n')

    relation_type_set_str = '{' + ', '.join(args.relation_type_set) + '}'

    demonstrations_formatted = []
    for demo_idx, demo in enumerate(icl_samples):
        demonstrations_formatted.append(f'Demo Index: {str(demo_idx)}\n')
        demonstrations_formatted.append(
            basic_icl_prompt.format(
                given_sentence=' '.join(demo["sentence"]),
                head_entity=demo["head_entity"]["span"],
                tail_entity=demo["tail_entity"]["span"],
                relation_type_set=relation_type_set_str,
                relation_type=demo["relation_type"],
                explanations=demo["explanations"],
            )
        )
    demonstrations_str = ''.join(demonstrations_formatted)
    demonstrations_str = demonstrations_str.replace('\n    ', '\n')

    formed_key_phase = get_formed_key_phase(head_entity=sample["head_entity"],
                                            tail_entity=sample["tail_entity"],
                                            given_sentence=sample["sentence"])

    original_scm_prompt = basic_instruction_prompt.format(
        relation_type_set=relation_type_set_str,
        demonstrations=demonstrations_str,
        given_sentence=' '.join(sample["sentence"]),
        head_entity=sample["head_entity"]["span"],
        tail_entity=sample["tail_entity"]["span"],
        formed_key_phase=formed_key_phase
    )

    return original_scm_prompt




def construct_hard_intervention_prompt(args, sample, icl_samples, specified_bias_hard_label=None):
    basic_instruction_prompt = '''Instruction: Given a sentence, explain why there is certain relation between the head and tail entities in the sentence.
    Demonstration:
    {demonstrations}
    <Hint> 
    Please learn the demonstration and follow the instruction, complete the "Reasoning Explanations" and "Prediction" parts of the new given instance. 
    Please end with <End of Instance> when complete the text.
    <Hint>
    <Start of Instance>
    Given Sentence: "{given_sentence}"
    Head Entity: "{head_entity}"
    Tail Entity: "{tail_entity}"
    The relation type between "{head_entity}" and "{tail_entity}" is "{relation_type}"
    Reasoning Explanations: {formed_key_phase}'''
    basic_instruction_prompt = basic_instruction_prompt.replace('\n    ', '\n')

    basic_icl_prompt = '''<Start of Instance>
    Given Sentence: "{given_sentence}"
    Head Entity: "{head_entity}"
    Tail Entity: "{tail_entity}"
    The relation type between "{head_entity}" and "{tail_entity}" is "{relation_type}"
    Reasoning Explanations: {explanations}
    Prediction: Given the sentence, the relation between the head entity "{head_entity}" and the tail entity "{tail_entity}" is "{relation_type}"
    <End of Instance>'''
    basic_icl_prompt = basic_icl_prompt.replace('\n    ', '\n')

    demonstrations_formatted = []
    for demo_idx, demo in enumerate(icl_samples):
        demonstrations_formatted.append(f'Demo Index: {str(demo_idx)}\n')
        demonstrations_formatted.append(
            basic_icl_prompt.format(
                given_sentence=' '.join(demo["sentence"]),
                head_entity=demo["head_entity"]["span"],
                tail_entity=demo["tail_entity"]["span"],
                relation_type=demo["relation_type"],
                explanations=demo["explanations"],
            )
        )
    demonstrations_str = ''.join(demonstrations_formatted)
    demonstrations_str = demonstrations_str.replace('\n    ', '\n')

    formed_key_phase = get_formed_key_phase(head_entity=sample["head_entity"],
                                            tail_entity=sample["tail_entity"],
                                            given_sentence=sample["sentence"])

    hard_label = sample["relation_type"]
    if specified_bias_hard_label != None:
        # 专门指定错误label时
        hard_label = specified_bias_hard_label

    hard_intervention_prompt = basic_instruction_prompt.format(
        demonstrations=demonstrations_str,
        given_sentence=' '.join(sample["sentence"]),
        head_entity=sample["head_entity"]["span"],
        tail_entity=sample["tail_entity"]["span"],
        relation_type=hard_label,
        formed_key_phase=formed_key_phase,
    )
    return hard_intervention_prompt


def construct_check_intervention_prompt(args, sample, intervention_x, icl_samples):
    basic_instruction_prompt = '''Instruction: Given a sentence and corresponding explanations, try to derive the relation label prediction.
    Demonstration:
    {demonstrations}
    <Hint> Please learn the demonstration and follow the instruction, output the inference result of the new given instance. <Hint>
    <Start of Instance>
    Given Sentence: "{given_sentence}"
    Relation Type Set: {relation_type_set}
    Head Entity: "{head_entity}"
    Tail Entity: "{tail_entity}"
    Reasoning Explanations: {explanations}
    Based on the above reasoning explanations, '''
    basic_instruction_prompt = basic_instruction_prompt.replace('\n    ', '\n')

    basic_icl_prompt = '''<Start of Instance>
    Given Sentence: "{given_sentence}"
    Relation Type Set: {relation_type_set}
    Head Entity: "{head_entity}"
    Tail Entity: "{tail_entity}"
    Reasoning Explanations: {explanations}
    Based on the above reasoning explanations, the relation between the head entity "{head_entity}" and the tail entity "{tail_entity}" is "{relation_type}"
    <End of Instance>'''

    basic_icl_prompt = basic_icl_prompt.replace('\n    ', '\n')

    relation_type_set_str = '{' + ', '.join(args.relation_type_set) + '}'

    demonstrations_formatted = []
    for demo_idx, demo in enumerate(icl_samples):
        demo_explanations = demo["explanations"]
        demo_relation_type = demo["relation_type"]

        filtered_demo_explanations = demo_explanations.replace(demo_relation_type, 'label')

        # if 'Therefore' in demo_explanations:
        #     filtered_demo_explanations = demo_explanations.split('Therefore')[0]
        # else:
        #     filtered_demo_explanations = demo_explanations.replace(demo_relation_type, 'label')

        demonstrations_formatted.append(f'Demo Index: {str(demo_idx)}\n')
        demonstrations_formatted.append(
            basic_icl_prompt.format(
                given_sentence=' '.join(demo["sentence"]),
                head_entity=demo["head_entity"]["span"],
                tail_entity=demo["tail_entity"]["span"],
                relation_type_set=relation_type_set_str,
                explanations=filtered_demo_explanations,
                relation_type=demo_relation_type,
            )
        )
    demonstrations_str = ''.join(demonstrations_formatted)
    demonstrations_str = demonstrations_str.replace('\n    ', '\n')

    filtered_intervention_x = intervention_x.replace(sample["relation_type"], 'label')
    soft_intervention_prompt = basic_instruction_prompt.format(
        demonstrations=demonstrations_str,
        given_sentence=' '.join(sample["sentence"]),
        relation_type_set=relation_type_set_str,
        head_entity=sample["head_entity"]["span"],
        tail_entity=sample["tail_entity"]["span"],
        explanations=filtered_intervention_x,
    )

    return soft_intervention_prompt


def parse_original_scm_output(args, output):
    pred_label, explanations = parse_reasoning_explanations_from_text(args=args,generated_text=output)
    original_scm = {
        "x": explanations,
        "pred_label": pred_label,
    }
    return original_scm


def parse_hard_intervention_output(args, output):
    _, hard_intervention_x = parse_reasoning_explanations_from_text(args=args,generated_text=output)
    return hard_intervention_x


def parse_soft_intervention_output(args, output):
    pred_label, _ = parse_reasoning_explanations_from_text(args=args,generated_text=output)

    return pred_label
