import csv

data_dir = "../../data/"

def gen_label():
    label_dict = {}
    i = 0
    with open(data_dir + "train_processed.csv", "r") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            label_dict[row[4]] = 2 * i
            label_dict[row[5]] = 2 * i + 1
            i += 1
    return label_dict