'''
script to convert the pruned data into simple passages.
'''

from data_prep import get_pruned_data, to_passage_simple


pruned_data, _ = get_pruned_data()
for pid, data in pruned_data.items():
    passage = to_passage_simple(data)
    with open(f'../data/simple_passages/{pid}.txt', 'w') as fp:
        fp.write(passage)
