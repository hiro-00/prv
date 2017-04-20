import csv
from collections import defaultdict

data_dir = "../data/"
model_dir = "../model/"

def create_corpas(row_num = 1000000):
    corpas = set()
    with open(data_dir + "train_processed.csv", "r") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        i = 0
        for row in reader:
            i += 1
            corpas.add(row[0])
            corpas.add(row[1])
            if i == row_num:
                break
    return list(corpas)

if __name__ == "__main__":
    corpas = create_corpas()
    with open("../data/corpas.txt", "w") as f:
        for sentence in corpas:
            print(sentence.strip("'\""), file = f)
            #print(sentence.strip("'"))