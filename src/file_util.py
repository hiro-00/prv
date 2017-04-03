import csv

data_dir = "../data/"
model_dir = "../model/"

def load_train_corpas(row_num = 1000000):
    document = []
    with open(data_dir + "train_processed.csv", "r") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        next(reader)
        i = 0
        for row in reader:
            i += 1
            document.append(row[4])
            document.append(row[5])
            if i == row_num:
                break
    return document