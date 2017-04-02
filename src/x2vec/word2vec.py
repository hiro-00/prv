import csv
import gensim

data_dir = "../../data/"
model_dir = "../../model/"

document = []
i = 0
with open(data_dir + "train_processed.csv", "r") as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    for row in reader:
        i+=1
        document.append(row[4].split())
        document.append(row[5].split())
        if i == 100:
            break

model = gensim.models.Word2Vec(document, min_count=1)

model.save(model_dir + "word2vec")
gensim.models.KeyedVectors.load(model_dir+"word2vec")