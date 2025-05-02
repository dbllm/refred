import json
import numpy as np
from typing import Dict, Collection
import pandas as pd
from pathlib import Path

# ======== general func =======

def load_npy(name, npy_path):
    res = []
    for item in np.load(npy_path):
        res.append({name: item})
    return res

def load_data(jsonl_path):
    '''
    load jsonl files to objs
    '''
    with open(jsonl_path, 'r') as rf:
        res = [json.loads(line) for line in rf]
        return res

def dump_data(jsonl_path, obj_list):
    pass

def merge_data(json_list):
    '''
    merge json objs
    '''
    assert all(len(x) == len(json_list[0]) for x in json_list), 'size does not match!'
    res = []
    for _tuple in zip(*json_list):
        new_json = {}
        for x in _tuple:
            for k, v in x.items():
                assert k not in new_json, 'conflict key [{}]!'.format(k)
                new_json[k] = v
        res.append(new_json)
    return res

def rename_data_column(data, name_mapping:Dict[str, str]):
    res = []
    for _tuple in data:
        for k,v in name_mapping.items():
            _tuple[v] = _tuple[k]
            del _tuple[k]
        res.append(_tuple)
    return res

def clean_data_columns(data, keep_cols: Collection[str]):
    res = []
    col_set = set(keep_cols)
    for _tuple in data:
        _new_t = [(k, v) for k,v in _tuple.items() if k in col_set]
        res.append(_new_t)
    return res


def flatten_items(json_obj, flatten_keys):
    new_list = []
    for item in json_obj:
        flatten_values = [item[k] for k in flatten_keys]
        remaining_keys = [k for k in item.keys() if k not in flatten_keys]
        assert all(len(x) == len(flatten_values[0]) for x in flatten_values), 'size does not match!'
        _remaining_items = dict([(k, item[k]) for k in remaining_keys])
        for i in range(len(flatten_values[0])):
            new_item = dict([(x, y[i]) for x, y in zip(flatten_keys, flatten_values)])
            new_item.update(_remaining_items)
            new_list.append(new_item)
    return new_list

# ------------ general data splits

def _split_data_index(total_num, percentage, shuffle=True):
    assert sum(percentage) == 1
    data = np.arange(total_num)
    if shuffle:
        np.random.shuffle(data)
    res = []
    prev_i =0
    n = len(data)
    for _p in np.cumsum(percentage):
        _current_p = round(_p*n)
        res.append(data[prev_i:_current_p])
        prev_i = _current_p
    return res

def split_data(data, percentage, random_seed=42, shuffle=True):
    np.random.seed(random_seed)
    assert sum(percentage) == 1
    index_arr = _split_data_index(len(data), percentage, shuffle)
    return [data.iloc[x] for x in index_arr]

# ============= specific func =====

def transform_pd_data(data: pd.DataFrame):
    '''
    Requires: id, answers, pred, total_prob, freq
    Produce: ans, gt, preds, probs, freqs
    '''
    mon_data = data
    mon_data.loc[:, 'ans'] = mon_data['answers'].apply(lambda x: x[0])
    unique_ans = sorted(mon_data['ans'].unique().tolist())
    
    ans_to_labels = dict([(_a, _i)for _i, _a in enumerate(unique_ans)])
    transformed_data = []
    for _idx, _row in data.iterrows():
        # compute probs.
        _unique_pred_dict = dict()
        for _pred, _prob, _freq in zip(_row['pred'], _row['total_prob'], _row['freq']):
            _pred_ans = _pred['response']
            if _pred_ans in _unique_pred_dict:
                _existing_one = _unique_pred_dict[_pred_ans]
                assert _prob == _existing_one['prob'] and _freq == _existing_one['freq']
            else:
                _unique_pred_dict[_pred_ans] = {
                    'ans': _pred_ans,
                    'prob': _prob,
                    'freq': _freq
                }
        # convert to arrays.
        label_dict_mapping = {}
        for _p in _unique_pred_dict.values():
            _key = None
            if _p['ans'] in ans_to_labels:
                _key = ans_to_labels[_p['ans']]
            else:
                _key = len(ans_to_labels)
            label_dict_mapping[_key] = (_p['prob'], _p['freq'])
        
        # get all labels, fill with 0s for other values.
        pred_labels = sorted(label_dict_mapping.keys())
        pred_probs = []
        pred_freqs = []
        for i in range(max(len(unique_ans), max(pred_labels)+1)):
            if i in label_dict_mapping:
                _prob, _freq = label_dict_mapping[i]
                pred_probs.append(_prob)
                pred_freqs.append(_freq)
            else:
                pred_probs.append(0)
                pred_freqs.append(0)

        # put back.
        transformed_data.append(
            (_row['id'], _row['ans'], ans_to_labels[_row['ans']], pred_labels, pred_probs, pred_freqs)
        )
    
    transformed_data = pd.DataFrame(transformed_data, columns=['id', 'ans', 'gt', 'preds', 'probs', 'freqs'])
    return transformed_data, unique_ans

# ====== specific instances =====

def load_llm_data(base_path: Path, with_sim=False):
    data = merge_data([
        load_data(base_path / 'all_raw.jsonl'),
        load_data(base_path / 'all_raw_scores.jsonl')
    ])
    data = pd.DataFrame(data)
    if not with_sim:
        return transform_pd_data(data)[0]
    data, ans_list = transform_pd_data(data)
    sim_matrix = np.zeros((len(ans_list), len(ans_list)))
    for i in range(len(ans_list)):
        sim_matrix[i,i] = 1
    res = np.stack([sim_matrix]* len(data)).tolist()
    data['pair_sim'] = res
    return data

def load_llm_data_embeddings(base_path: Path, embedding_name: str, with_sim=False):
    data = load_llm_data(base_path, with_sim)
    embeddings = np.load(base_path / embedding_name)
    data['embedding'] = embeddings.tolist()
    return data


if __name__ == '__main__':
    # res = merge_data([
    #     [{'a': 'a'}], [{'b': 'b', "c": 'c'}]
    # ])
    # print(res)

    res = flatten_items(
        [{'a_list': [1, 2], 'b_list': ['a', 'b'], 'c_list': [1, 2,3]}], 
        ['a_list', 'b_list']
    )
    print(res)

    data = pd.DataFrame(res)
    print(data)

    embed = np.random.rand(2, 3)
    data['embed'] = embed.tolist()
    print(embed)
    print(data)

    print(type(data.iloc[0]['embed']))

    revert_embed = np.array(data['embed'].to_list())

    print(type(revert_embed))