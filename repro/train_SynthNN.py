import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import re
import pymatgen as mg
import tensorflow as tf
import random
import os
from pymatgen.core.periodic_table import Element
import linecache

POSITIVE_SAMPLE_FILE = "repro/icsd_positive.txt"
NEGATIVE_SAMPLE_FILE = "repro/standard_neg_ex_tr_val_v5_balanced_shuffled.txt"
OUTPUT_DIR = "repro/training_output/"
# hyperparameters taken from 2020_best run
MODEL_HYPERPARAMETERS = [1, 4, 1, 1, 2]

# these select hyperparameters from the lists below:
randint = MODEL_HYPERPARAMETERS
tstep = [1e-2, 5e-3, 2e-3, 5e-4, 2e-4][randint[0]]
no_h1 = [30, 40, 50, 60, 80][randint[1]]
no_h2 = [30, 40, 50, 60, 80][randint[2]]
batch_size = [512, 512, 1024, 1024, 1024][randint[3]]
# steps in supervised stage
semi_starting = [20000, 40000, 60000, 80000, 100000][randint[4]]

# steps in semi-supervised stage
num_steps = 800000

def load_data(data, is_charge_balanced, max_atoms=5, max_coefficient=100000):
    # takes input file (icsd_full_properties_no_frac_charges) and processes the data and applies some filters
    output_array = []
    coeff_array = np.zeros((10000, 1))
    element_names_array = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
    ]
    for i in range(len(data)):
        try:
            comp = mg.core.composition.Composition(data[i, 1])
        except:
            continue  # bad formula
        if len(mg.core.composition.Composition(data[i, 1])) == 1:
            continue
        truth_array = []
        for element_name in (
            mg.core.composition.Composition(data[i, 1]).as_dict().keys()
        ):
            if element_name not in element_names_array:
                truth_array.append("False")
        if "False" in truth_array:
            continue
        if is_charge_balanced:
            if "True" in data[i][8]:
                if len(mg.core.composition.Composition(data[i, 1])) < max_atoms:
                    values = (
                        mg.core.composition.Composition(data[i, 1]).as_dict().values()
                    )
                    for value in values:
                        coeff_array[int(value)] = coeff_array[int(value)] + 1
                    large_values = [x for x in values if x > max_coefficient]
                    if len(large_values) == 0:
                        output_array.append(
                            mg.core.composition.Composition(
                                data[i, 1]
                            ).alphabetical_formula.replace(" ", "")
                        )
        else:
            output_array.append(
                mg.core.composition.Composition(
                    data[i, 1]
                ).alphabetical_formula.replace(" ", "")
            )
    return np.unique(output_array)


def get_features(data0):
    p = re.compile("[A-Z][a-z]?\d*\.?\d*")
    p3 = re.compile("[A-Z][a-z]?")
    p5 = re.compile("\d+\.?\d+|\d+")
    data0_ratio = []
    for i in data0:

        x = mg.core.composition.Composition(i).alphabetical_formula.replace(" ", "")
        p2 = p.findall(x)
        temp1, temp2 = [], []
        for x in p2:
            temp1.append(Element[p3.findall(x)[0]].number)
            kkk = p5.findall(x)
            if len(kkk) < 1:
                temp2.append(1)
            else:
                temp2.append(kkk[0])
        data0_ratio.append([temp1, list(map(float, temp2))])

    I = 94
    featmat0 = np.zeros((len(data0_ratio), I))
    # featmat: n-hot vectors with fractions
    for idx, ent in enumerate(data0_ratio):
        for idy, at in enumerate(ent[0]):
            featmat0[idx, at - 1] = ent[1][idy] / sum(ent[1])
    return featmat0


def random_lines(filename, file_size, num_samples):
    idxs = random.sample(range(1, file_size), num_samples)
    return ([linecache.getline(filename, i) for i in idxs], idxs)


def get_batch_val(neg_positive_ratio):
    random.seed(3)
    np.random.seed(3)
    noTr_positives = 29000  # number positive examples in train set
    noTr_negatives = (
        noTr_positives * neg_positive_ratio
    )  # no. negatives examples in train set
    noTr = noTr_positives + (noTr_negatives)  # total size of train set
    # only sample from first 90% of dataset ( need to shuffle first because GNN_icsd is alphabetical!)
    data1 = []
    f = open(POSITIVE_SAMPLE_FILE)
    i = 0
    for line in f:
        if i > noTr_positives and i < noTr_positives * 1.05:
            data1.append(line.replace("\n", ""))
        i += 1
    f.close()

    data0 = []
    f = open(NEGATIVE_SAMPLE_FILE)
    i = 0
    for line in f:
        if i > noTr_negatives and i < noTr_negatives * 1.05:
            data0.append(line.replace("\n", ""))
        i += 1
    f.close()
    # shuffle the positive and negative examples with themselves
    negative_indices = list(range(0, len(data0)))
    random.shuffle(negative_indices)
    positive_indices = list(range(0, len(data1)))
    random.shuffle(positive_indices)
    data0 = np.array(data0)
    data1 = np.array(data1)

    data0 = data0[negative_indices]
    data1 = data1[positive_indices]
    featmat0 = get_features(data0)
    featmat1 = get_features(data1)

    # get labels
    labs = np.zeros((len(data0) + len(data1), 1))
    for ind, ent in enumerate(data1):
        labs[ind, 0] = 1
    unique, counts = np.unique(labs, return_counts=True)
    ind0 = np.where(labs == 0)[0]  # indices of label=0
    ind1 = np.where(labs == 1)[0]  # indices of label=1
    # print(len(ind0),len(ind1))

    # combine positives and negatives and shuffle

    featmat3 = np.concatenate(
        (featmat0, featmat1)
    )  # set legths of labels 0 and 1 to be the same in the new feature matrix featmat3
    datasorted = np.concatenate((data0, data1))  # data ordered the same as featmat3
    labs3 = np.concatenate(
        (labs[ind0], labs[ind1]), axis=0
    )  # labels ordered the same as featmat3

    noS = len(featmat3)
    ind = list(range(0, noS))  # training set index
    random.shuffle(ind)  # shuffle training set index
    # indB = list(range(0,noTr_subset)) #used later for batch
    labs3 = np.column_stack((labs3, np.abs(labs3 - 1)))

    xtr_batch = featmat3[ind[0:], :]
    ytr_batch = labs3[ind[0:], :]
    data_batch = datasorted[ind[0:]]
    return (xtr_batch, ytr_batch, data_batch)


def get_batch(
    batch_size,
    neg_positive_ratio,
    use_semi_weights,
    model_name,
    seed=False,
    seed_value=0,
):
    def random_lines(filename, file_size, num_samples):
        idxs = random.sample(range(1, file_size), num_samples)
        return ([linecache.getline(filename, i) for i in idxs], idxs)

    if seed:
        random.seed(seed_value)
        np.random.seed(seed_value)
    else:
        random.seed()
        np.random.seed()
    num_positive_examples = int(np.floor(batch_size * (1 / (1 + neg_positive_ratio))))
    num_negative_examples = batch_size - num_positive_examples
    noTr_positives = 29000  # number positive examples in train set
    noTr_negatives = (
        noTr_positives * neg_positive_ratio
    )  # no. negatives examples in train set
    noTr = noTr_positives + (noTr_negatives)  # total size of train set
    # only sample from first 90% of dataset ( need to shuffle first because GNN_icsd is alphabetical!)
    data1 = []
    pulled_lines1, idxs1 = random_lines(
        POSITIVE_SAMPLE_FILE,
        noTr_positives,
        num_positive_examples,
    )
    for line in pulled_lines1:
        data1.append(line.replace("\n", ""))
    data0 = []
    pulled_lines0, idxs0 = random_lines(
        NEGATIVE_SAMPLE_FILE,
        noTr_negatives,
        num_negative_examples,
    )
    for line in pulled_lines0:
        data0.append(line.replace("\n", ""))

    # do consistent shuffling once examples have been chosen
    random.seed(3)
    np.random.seed(3)
    # shuffle the positive and negative examples with themselves
    negative_indices = list(range(0, len(data0)))
    random.shuffle(negative_indices)
    positive_indices = list(range(0, len(data1)))
    random.shuffle(positive_indices)
    data0 = np.array(data0)
    data1 = np.array(data1)
    idxs0 = np.array(idxs0)
    idxs1 = np.array(idxs1)
    data0 = data0[negative_indices]
    data1 = data1[positive_indices]
    featmat0 = get_features(data0)
    featmat1 = get_features(data1)
    idxs0 = idxs0[negative_indices]
    idxs1 = idxs1[positive_indices]

    # get labels
    labs = np.zeros((len(data0) + len(data1), 1))
    for ind, ent in enumerate(data1):
        labs[ind, 0] = 1
    unique, counts = np.unique(labs, return_counts=True)
    ind0 = np.where(labs == 0)[0]  # indices of label=0
    ind1 = np.where(labs == 1)[0]  # indices of label=1
    # print(len(ind0),len(ind1))

    # combine positives and negatives and shuffle
    featmat3 = np.concatenate(
        (featmat0, featmat1)
    )  # set legths of labels 0 and 1 to be the same in the new feature matrix featmat3
    datasorted = np.concatenate((data0, data1))  # data ordered the same as featmat3
    labs3 = np.concatenate(
        (labs[ind0], labs[ind1]), axis=0
    )  # labels ordered the same as featmat3
    idxs_full = np.concatenate((idxs0, idxs1))
    noS = len(featmat3)
    ind = list(range(0, noS))  # training set index
    random.shuffle(ind)  # shuffle training set index
    labs3 = np.column_stack((labs3, np.abs(labs3 - 1)))
    xtr_batch = featmat3[ind[0:], :]
    ytr_batch = labs3[ind[0:], :]
    data_batch = datasorted[ind[0:]]
    idxs_full = idxs_full[ind[0:]]
    # all weights stuff here
    weights_full = []
    if use_semi_weights:
        weights1 = []
        file = open(OUTPUT_DIR + "semi_weights_testing_pos_30M" + model_name + ".txt", "r")
        content = file.readlines()
        weights1 = []
        for i in idxs1:
            weights1.append(float(content[i - 1].split()[1]))
        file.close()

        weights0 = []
        file = open(OUTPUT_DIR + "semi_weights_testing_neg_30M" + model_name + ".txt", "r")
        content = file.readlines()
        for i in idxs0:
            weights0.append(float(content[i - 1].split()[1]))
        file.close()
        weights0 = np.array(weights0)
        weights1 = np.array(weights1)
        weights0 = weights0[negative_indices]
        weights1 = weights1[positive_indices]
        weights_full = np.concatenate((weights0, weights1))
        weights_full = weights_full[ind[0:]]
    else:
        weights_full = np.ones(len(idxs_full))
    return (xtr_batch, ytr_batch, data_batch, weights_full, idxs_full)


def perf_measure(y_actual, y_hat, cutoff=0.5):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == 1 and y_hat[i] > cutoff:
            TP += 1
        if y_hat[i] > cutoff and y_actual[i] == 0:
            FP += 1
        if y_actual[i] == 0 and y_hat[i] < cutoff:
            TN += 1
        if y_hat[i] < cutoff and y_actual[i] == 1:
            FN += 1

    return (TP, FP, TN, FN)


def SemiTransform(Xtrain, Ytrain, probtrain):
    X1 = Xtrain[np.where(Ytrain == 1)[0]]
    X0 = Xtrain[np.where(Ytrain == 0)[0]]
    Xsemi = np.row_stack((X1, X0, X0))
    prob0 = probtrain[np.where(Ytrain == 0)[0]]
    Y1 = Ytrain[np.where(Ytrain == 1)[0]]
    Y0 = Ytrain[np.where(Ytrain == 0)[0]]
    Ysemi = np.concatenate((Y1, Y0, Y0 + 1))

    # c=np.mean(probtrain[Ytrain == 1][:,1])
    c = np.max(probtrain[np.where(Ytrain == 1)[0]])
    p = prob0
    w = p / (1 - p)
    w *= (1 - c) / c
    weights = np.ones(len(Ysemi))
    weights[len(Y1) : len(Y1) + len(Y0)] = 1 - w
    weights[len(Y1) + len(Y0) :] = w
    Ysemi = np.column_stack((Ysemi, np.abs(Ysemi - 1)))

    return Xsemi, Ysemi, weights


# hyperparameters
element_names_array = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
]

np.random.seed()
random.seed()
M = 30
tf.compat.v1.disable_eager_execution()
# randint = np.random.randint(0, 5, 5)
# randint = MODEL_HYPERPARAMETERS
# tstep = [1e-2, 5e-3, 2e-3, 5e-4, 2e-4][randint[0]]
# num_steps = 800000
# no_h1 = [30, 40, 50, 60, 80][randint[1]]
# no_h2 = [30, 40, 50, 60, 80][randint[2]]
# batch_size = [512, 512, 1024, 1024, 1024][randint[3]]
# semi_starting = [20000, 40000, 60000, 80000, 100000][randint[4]]

neg_pos_ratio = 20
weight_for_0 = (1 + neg_pos_ratio) / (2 * neg_pos_ratio)
weight_for_1 = (1 + neg_pos_ratio) / (2 * 1)

model_name_params = (
    str(randint[0])
    + str(randint[1])
    + str(randint[2])
    + str(randint[3])
    + str(randint[4])
)

xtr, ytr, batch_data, weights, idxs = get_batch(
    batch_size, neg_pos_ratio, use_semi_weights=False, model_name=model_name_params
)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, xtr.shape[1]])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])

W1 = tf.Variable(
    tf.random.truncated_normal([xtr.shape[1], M], 0, 3)
)  # shape, mean, std
F1 = tf.Variable(tf.random.truncated_normal([M, no_h1], 0, 1))
F2 = tf.Variable(tf.random.truncated_normal([no_h1, no_h2], 0, 1))
F3 = tf.Variable(tf.random.truncated_normal([no_h2, 2], 0, 1))
b1 = tf.Variable(tf.random.truncated_normal([no_h1], 0, 1))
b2 = tf.Variable(tf.random.truncated_normal([no_h2], 0, 1))
b3 = tf.Variable(tf.random.truncated_normal([2], 0, 1))

semi_weights = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.initialize_all_variables())

z0_raw = tf.multiply(tf.expand_dims(x, 2), tf.expand_dims(W1, 0))  # (ntr, I, M)
tempmean, var = tf.nn.moments(x=z0_raw, axes=[1])
z0 = tf.concat([tf.reduce_sum(input_tensor=z0_raw, axis=1)], 1)  # (ntr, M)
z1 = tf.add(tf.matmul(z0, F1), b1)  # (ntr, no_h1)
a1 = tf.tanh(z1)  # (ntr, no_h1)
z2 = tf.add(tf.matmul(a1, F2), b2)  # (ntr,no_h1)
a2 = tf.tanh(z2)  # (ntr, no_h1)
z3 = tf.add(tf.matmul(a2, F3), b3)  # (ntr, 2)
a3 = tf.nn.softmax(z3)  # (ntr, 2)
clipped_y = tf.clip_by_value(a3, 1e-10, 1.0)
cross_entropy = -tf.reduce_sum(
    input_tensor=tf.multiply(
        y_ * tf.math.log(clipped_y) * np.array([weight_for_1, weight_for_0]),
        semi_weights,
    )
)
correct_prediction = tf.equal(tf.argmax(input=a3, axis=1), tf.argmax(input=y_, axis=1))
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
performance_array = []
loss_array = []
element_sum = np.zeros((94, 1))
epoch_counter = 0
xtr_new = xtr
ytr_new = ytr

train_step = tf.compat.v1.train.AdamOptimizer(tstep).minimize(cross_entropy)
sess.run(tf.compat.v1.initialize_all_variables())
full_weights = np.ones((len(ytr), 1))
best_perf = 0
xval, yval, val_data = get_batch_val(neg_pos_ratio)

W1_val = []
F1_val = []
F2_val = []
F3_val = []
b1_val = []
b2_val = []
b3_val = []
print("Model Initial Training")

# change to semi_starting
for i in range(semi_starting):
    epoch_counter = epoch_counter + 1
    batchx, batchy, batch_data, weights, idxs = get_batch(
        batch_size, neg_pos_ratio, use_semi_weights=False, model_name=model_name_params
    )
    indB = list(range(0, len(xtr_new)))
    random.shuffle(indB)
    current_weights = full_weights[indB[0:batch_size], :]
    train_step.run(feed_dict={x: batchx, y_: batchy, semi_weights: current_weights})
    if i % 100 == 0:
        print("step: ", i)
    if i % 1000 == 0:
        preds = a3.eval(feed_dict={x: xval, y_: yval, semi_weights: full_weights})
        TP, FP, TN, FN = perf_measure(np.array(yval)[:, 0], np.array(preds)[:, 0])
        val_accuracy = accuracy.eval(
            feed_dict={x: xval, y_: yval, semi_weights: current_weights}
        )
        train_accuracy = accuracy.eval(
            feed_dict={x: batchx, y_: batchy, semi_weights: current_weights}
        )
        performance_array.append([train_accuracy, val_accuracy, TP, FP, TN, FN])
        print(i)
        print([train_accuracy, val_accuracy, TP, FP, TN, FN])
        # np.savetxt('performance_matrix_TL_v3_' + str(randint[0]) + str(randint[1]) + str(randint[2]) + str(randint[3]) + '.txt',performance_array, fmt='%s')
        if val_accuracy > best_perf:
            best_perf = val_accuracy
            W1_val = sess.run(W1)
            F1_val = sess.run(F1)
            F2_val = sess.run(F2)
            F3_val = sess.run(F3)
            b1_val = sess.run(b1)
            b2_val = sess.run(b2)
            b3_val = sess.run(b3)

# print out all preds to a file (for weighting for semi-supervised learning)
print("Model Testing on Positives")
file_output = open(OUTPUT_DIR + "semi_weights_testing_pos_30M" + model_name_params + ".txt", "a")
file_positives = open(POSITIVE_SAMPLE_FILE, "r")
Lines = file_positives.readlines()
for line in Lines:
    xtr = get_features([line.replace("\n", "")])
    ytr = [[0, 1]]
    pred = a3.eval(
        feed_dict={
            x: xtr,
            y_: ytr,
            semi_weights: current_weights,
            W1: W1_val,
            F1: F1_val,
            F2: F2_val,
            F3: F3_val,
            b1: b1_val,
            b2: b2_val,
            b3: b3_val,
        }
    )
    file_output.write(line.replace("\n", "") + " " + str(pred[0][0]) + "\n")
file_positives.close()
file_output.close()

print("Model Testing on Negatives")
file_output = open(OUTPUT_DIR + "semi_weights_testing_neg_30M" + model_name_params + ".txt", "a")
file_negatives = open(NEGATIVE_SAMPLE_FILE, "r")
Lines = file_negatives.readlines()
for line in Lines:
    xtr = get_features([line.replace("\n", "")])
    ytr = [[0, 1]]
    pred = a3.eval(
        feed_dict={
            x: xtr,
            y_: ytr,
            semi_weights: current_weights,
            W1: W1_val,
            F1: F1_val,
            F2: F2_val,
            F3: F3_val,
            b1: b1_val,
            b2: b2_val,
            b3: b3_val,
        }
    )
    file_output.write(line.replace("\n", "") + " " + str(pred[0][0]) + "\n")
file_negatives.close()
file_output.close()
sess.close()

print("Doing Semi-supervised Learning")
np.random.seed()
random.seed()
M = 30
x = tf.compat.v1.placeholder(tf.float32, shape=[None, xtr.shape[1]])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])

W1 = tf.Variable(
    tf.random.truncated_normal([xtr.shape[1], M], 0, 3)
)  # shape, mean, std
F1 = tf.Variable(tf.random.truncated_normal([M, no_h1], 0, 1))
F2 = tf.Variable(tf.random.truncated_normal([no_h1, no_h2], 0, 1))
F3 = tf.Variable(tf.random.truncated_normal([no_h2, 2], 0, 1))
b1 = tf.Variable(tf.random.truncated_normal([no_h1], 0, 1))
b2 = tf.Variable(tf.random.truncated_normal([no_h2], 0, 1))
b3 = tf.Variable(tf.random.truncated_normal([2], 0, 1))
semi_weights = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.initialize_all_variables())

z0_raw = tf.multiply(tf.expand_dims(x, 2), tf.expand_dims(W1, 0))  # (ntr, I, M)
tempmean, var = tf.nn.moments(x=z0_raw, axes=[1])
z0 = tf.concat([tf.reduce_sum(input_tensor=z0_raw, axis=1)], 1)  # (ntr, M)
z1 = tf.add(tf.matmul(z0, F1), b1)  # (ntr, no_h1)
a1 = tf.tanh(z1)  # (ntr, no_h1)
z2 = tf.add(tf.matmul(a1, F2), b2)  # (ntr,no_h1)
a2 = tf.tanh(z2)  # (ntr, no_h1)
z3 = tf.add(tf.matmul(a2, F3), b3)  # (ntr, 2)
a3 = tf.nn.softmax(z3)  # (ntr, 2)

clipped_y = tf.clip_by_value(a3, 1e-10, 1.0)
cross_entropy = -tf.reduce_sum(input_tensor=tf.multiply(y_ * tf.math.log(clipped_y), semi_weights))
correct_prediction = tf.equal(tf.argmax(input=a3, axis=1), tf.argmax(input=y_, axis=1))
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
performance_array = []
loss_array = []
element_sum = np.zeros((94, 1))
epoch_counter = 0
best_perf = 0
train_step = tf.compat.v1.train.AdamOptimizer(tstep).minimize(cross_entropy)
sess.run(tf.compat.v1.initialize_all_variables())

# change to num_steps
for i in range(num_steps):
    epoch_counter = epoch_counter + 1
    batchx, batchy, batch_data, weights, idxs = get_batch(
        batch_size, neg_pos_ratio, use_semi_weights=True, model_name=model_name_params
    )
    weights = np.reshape(weights, [len(weights), 1])
    train_step.run(feed_dict={x: batchx, y_: batchy, semi_weights: weights})
    if i % 100 == 0:
        print("step: ", i)
    # loss_array.append([])
    if i % 1000 == 0:
        preds = a3.eval(feed_dict={x: xval, y_: yval, semi_weights: weights})
        TP, FP, TN, FN = perf_measure(np.array(yval)[:, 0], np.array(preds)[:, 0])
        val_accuracy = accuracy.eval(
            feed_dict={x: xval, y_: yval, semi_weights: weights}
        )
        train_accuracy = accuracy.eval(
            feed_dict={x: batchx, y_: batchy, semi_weights: weights}
        )
        performance_array.append([train_accuracy, val_accuracy, TP, FP, TN, FN])
        print([train_accuracy, val_accuracy, TP, FP, TN, FN])
        np.savetxt(
            OUTPUT_DIR + "performance_matrix_TL_v3_30M_" + model_name_params + ".txt",
            performance_array,
            fmt="%s",
        )

    if i % 1000 == 0:
        if val_accuracy > best_perf:
            model_name = "30M_synth_v3_semi" + model_name_params + ".txt"
            best_perf = val_accuracy
            W1_val = sess.run(W1)
            F1_val = sess.run(F1)
            F2_val = sess.run(F2)
            F3_val = sess.run(F3)
            b1_val = sess.run(b1)
            b2_val = sess.run(b2)
            b3_val = sess.run(b3)
            np.savetxt(OUTPUT_DIR + "W1_" + model_name, W1_val)
            np.savetxt(OUTPUT_DIR + "F1_" + model_name, F1_val)
            np.savetxt(OUTPUT_DIR + "F2_" + model_name, F2_val)
            np.savetxt(OUTPUT_DIR + "F3_" + model_name, F3_val)
            np.savetxt(OUTPUT_DIR + "b1_" + model_name, b1_val)
            np.savetxt(OUTPUT_DIR + "b2_" + model_name, b2_val)
            np.savetxt(OUTPUT_DIR + "b3_" + model_name, b3_val)
sess.close()
