from torch.utils.data.dataset import Dataset
import pickle
from torch.utils.data import DataLoader
import json
# import process_utils
import numpy as np

class PoemDataset(Dataset):

    def __init__(self, data_path):

        super(PoemDataset, self).__init__()
        with open(data_path, 'rb') as fp:
            self.cases = pickle.load(fp)
        print(len(self.cases))

    def __len__(self):
        return len(self.cases)

    # get one sample by indexing
    def __getitem__(self, index):

        return self.cases[index][0].astype(np.int64), \
               self.cases[index][1].astype(np.int64), \
               self.cases[index][2].astype(np.int64)



if __name__ == '__main__':

    import process_utils
    training_params = {"batch_size": 32,
                       "shuffle": True,
                       "drop_last": True}

    training_set = PoemDataset('../model/train.pkl')
    training_generator = DataLoader(training_set, **training_params)

    with open('../model/phrase_rank.json', 'r', encoding='utf-8') as fp:
        phrase_rank = json.load(fp)
    with open('../model/token2id.json', 'r', encoding='utf-8') as fp:
        token2id = json.load(fp)

    dg = process_utils.DatasetGen(token2id, phrase_rank)

    for iter, (keywords, prceeding_texts, current_lines) in enumerate(training_generator):

        print(keywords.shape)
        for i in range(len(keywords)):

            keyword = dg.recover(keywords[i].cpu().numpy())
            preceeding_text = dg.recover(prceeding_texts[i].cpu().numpy())
            current_line = dg.recover(current_lines[i].cpu().numpy())

            print(keyword)
            print(preceeding_text)
            print(current_line)
            print('===============')
        exit(0)




