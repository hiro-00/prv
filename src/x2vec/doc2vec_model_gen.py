import csv
from gensim.models.doc2vec import TaggedDocument
import gensim
from src.x2vec.doc2vec_util import gen_label

data_dir = "../../data/"
model_dir = "../../model/"


def doc2vec():
    label_dict = gen_label()
    document = []
    i = 0
    with open(data_dir + "train_processed.csv", "r") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            i+=1
            document.append(TaggedDocument(words = row[4].split(), tags = [label_dict[row[4]]]))
            document.append(TaggedDocument(words = row[5].split(), tags = [label_dict[row[5]]]))
            if i == 30:
                pass

    model = gensim.models.Doc2Vec(document, min_count=1)

    model.save(model_dir + "doc2vec")

doc2vec()