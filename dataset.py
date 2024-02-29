import json
import random
from collections.abc import Mapping

import torch
from tokenizers import Encoding
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from xztrainer import ContextType


def merge_vacancy(x):
    txt = f'{x["name"]}\n{x["description"]}'
    if x['keywords']:
        txt = f'{txt}\n{x["keywords"]}'
    if x['comment']:
        txt = f'{txt}\n{x["comment"]}'
    return txt


def merge_exp(exp):
    employer = exp['employer'] or 'Неизвестно'
    position = exp['position'] or 'Неизвестно'
    city = exp['city'] or 'Город неизвестен'
    description = exp['description'] or ''
    return f'Компания {employer}, {city}\nДолжность: {position}\nЧто делал: {description}'


def default_exp():
    return f'Компания Нет\nДолжность: Безработный\nЧто делал: неизвестно'


def merge_candidate_general_info(x):
    country = x['country'] or 'Страна неизвестна'
    city = x['city'] or 'Город неизвестен'
    about = x['about'] or '?'
    key_skills = x['key_skills'] or '?'
    return f'Место жительства: {country}, {city}\nО себе: {about}\nРаботал с: {key_skills}'


def split_data_train(data):
    lst = []
    for vac_i, vac in enumerate(data):
        for _ in range(10):
            for resume_ok in vac['confirmed_resumes']:
                curr_lst = []
                curr_lst.append({
                    'vacancy': vac['vacancy'],
                    'resume': resume_ok,
                    'target': 1
                })
                for resume_fail in random.sample(vac['failed_resumes'], min(5, len(vac['failed_resumes']))):
                    curr_lst.append({
                        'vacancy': vac['vacancy'],
                        'resume': resume_fail,
                        'target': 0
                    })
                # dc = data.copy()
                # dc.pop(vac_i)
                # other_vacs = random.choices(dc, k=2)
                # other_vacs_resumes = [y  for x in other_vacs for y in x['failed_resumes'] + x['confirmed_resumes']]
                # other_vacs_resumes_choose = random.choices(other_vacs_resumes, k=3)
                # for r in other_vacs_resumes_choose:
                #     curr_lst.append(
                #         {'vacancy': vac['vacancy'],
                #          'resume': r,
                #          'target': 0}
                #     )
                lst.append(curr_lst)
    return lst


def split_data_eval(data):
    lst = []
    for i, vac in enumerate(data):
        for resume_ok in vac['confirmed_resumes']:
            lst.append({
                'vacancy': vac['vacancy'],
                'resume': resume_ok,
                'target': 1,
                'class': i,
                'id': resume_ok['uuid']
            })
        for resume_fail in vac['failed_resumes']:
            lst.append({
                'vacancy': vac['vacancy'],
                'resume': resume_fail,
                'target': 0,
                'class': i,
                'id': resume_fail['uuid']
            })
    return lst


def split_data_infer(data):
    lst = []
    for i, vac in enumerate(data):
        for resume in vac['resumes']:
            lst.append({
                'vacancy': vac['vacancy'],
                'resume': resume,
                'class': i,
                'id': resume['uuid']
            })
    return lst


def prepend_passage(txt: str):
    return f'passage: {txt}'


def encoding_to_tensors(x: Encoding):
    return {
        'input_ids': torch.tensor(x.ids, dtype=torch.long),
        'attention_mask': torch.tensor(x.attention_mask, dtype=torch.long)
    }


class QFormerDataset(Dataset):
    def __init__(self, data, mode):
        self.train = mode == ContextType.TRAIN
        if mode == ContextType.TRAIN:
            self.data = split_data_train(data)
        elif mode == ContextType.EVAL:
            self.data = split_data_eval(data)
        else:
            self.data = split_data_infer(data)

        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')

    def __len__(self):
        return len(self.data)

    def _element(self, el):
        vac_txt = merge_vacancy(el['vacancy'])
        cand_txt = merge_candidate_general_info(el['resume'])
        exp_txts = []
        for exp in el['resume'].get('experienceItem', []):
            exp_txts.append(merge_exp(exp))

        if len(exp_txts) == 0:
            exp_txts.append(default_exp())

        vac_txt = prepend_passage(vac_txt)
        cand_txt = prepend_passage(cand_txt)
        exp_txts = [prepend_passage(x) for x in exp_txts]

        vac_enc = self.tokenizer(vac_txt, max_length=512)
        cand_enc = self.tokenizer(cand_txt, max_length=512)
        exp_encs = self.tokenizer(exp_txts, max_length=512)

        dct = {
            'vacancy': encoding_to_tensors(vac_enc.encodings[0]),
            'candidate': encoding_to_tensors(cand_enc.encodings[0]),
            'experiences': [encoding_to_tensors(x) for x in exp_encs.encodings],
        }
        if 'target' in el:
            dct['target'] = torch.scalar_tensor(el['target'] == 1, dtype=torch.bool)
        if 'class' in el:
            dct['class'] = torch.scalar_tensor(el['class'], dtype=torch.long)
        if 'id' in el:
            dct['id'] = el['id']
        return dct

    def __getitem__(self, item):
        els = self.data[item]
        if self.train:
            outs = []
            for el in els:
                outs.append(self._element(el))
            return outs
        else:
            return self._element(els)


def stack_deep(x):
    if isinstance(x, dict):
        return {k: stack_deep(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [stack_deep(y) for y in x]
    elif isinstance(x, str):
        return [x]
    else:
        return x.unsqueeze(0)


class Collator():
    def __call__(self, batch):
        return stack_deep(batch[0])

