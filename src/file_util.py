import csv

data_dir = "data/"
model_dir = "../model/"

def load_train_corpas(row_num = 1000000):
    document = []
    with open(data_dir + "corpas.txt", "r") as f:
        i = 0
        for row in f:
            i += 1
            document.append(row.rstrip())
            if i == row_num:
                break
    return document

if __name__ == "__main__":
    for line in load_train_corpas():
        print(line)