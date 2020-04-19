import math
import random
import matplotlib.pyplot as plt


def reduce_dataset(feature, value, dataset):
    inds = set(idx for idx, val in enumerate(dataset[feature]) if val == value)
    ret = {x: [] for x in dataset}
    for feat in dataset:
        for idx, val in enumerate(dataset[feat]):
            if idx in inds:
                ret[feat].append(val)
    return ret


def entropy(feature_vector):
    set_features = set(feature_vector)
    prob = {x: feature_vector.count(x)/len(feature_vector) for x in set_features}
    return -sum(prob[x] * math.log(prob[x], 2) for x in prob)


def choose_feature(dataset):
    orig_entropy = entropy(dataset[label])
    best_feature = ''
    min_entropy = float('inf')
    for feature, vals_list in dataset.items():
        if feature == label:
            continue
        set_vals = set(vals_list)
        prob = {x: vals_list.count(x) / len(vals_list) for x in set_vals}
        exp_ent = 0
        for val in set_vals:
            curr_reduced = reduce_dataset(feature, val, dataset)
            exp_ent += entropy(curr_reduced[label]) * prob[val]
        # print(feature, ": ", entropy(data[label]) - exp_ent)
        if exp_ent < min_entropy:
            min_entropy = exp_ent
            best_feature = feature
    return best_feature, orig_entropy - min_entropy


def make_decision_tree(dataset, curr_dict, depth=0):
    best_feature, info_gain = choose_feature(dataset)
    pr_tree.append(("%s? (information gain: %s)" % (best_feature, info_gain), depth))
    curr_dict[best_feature] = {}
    for val in set(dataset[best_feature]):
        red_dataset = reduce_dataset(best_feature, val, dataset)
        red_ent = entropy(red_dataset[label])
        if red_ent == 0:
            pr_tree.append(("%s --> %s" % (val, red_dataset[label][0]), depth + 1))
            curr_dict[best_feature][val] = red_dataset[label][0]
        else:
            pr_tree.append(("%s (with current entropy %s)" % (val, red_ent), depth + 1))
            curr_dict[best_feature][val] = make_decision_tree(red_dataset, {}, depth=depth + 2)
    return curr_dict


def print_tree(dataset, dec_tree):
    print(" * Starting Entropy:", entropy(dataset[label]))
    for element, depth in dec_tree:
        print("  " * depth, "*", element)


def split_train_test(dataset, test_set_size=50):
    inds = set(random.sample(range(len(dataset[label])), test_set_size))
    trn, tst = {x: [] for x in dataset}, {x: [] for x in dataset}
    for feat in dataset:
        for idx, val in enumerate(dataset[feat]):
            if idx in inds:
                tst[feat].append(val)
            else:
                trn[feat].append(val)
    return trn, tst


def apply_decisions(test_set, ind, dec_tree):
    if type(dec_tree) == str:
        return dec_tree
    for feat in dec_tree.keys():
        return apply_decisions(test_set, ind, dec_tree[feat][test_set[feat][ind]])


def accuracy(test_set, dec_tree):
    # returns accuracy as a percent
    correct = 0
    for ind in range(len(test_set[label])):
        if test_set[label][ind] == apply_decisions(test_set, ind, dec_tree):
            correct += 1
    return 100 * correct / len(test_set[label])


data = {}
NONMISSING = {}
index_to_feature = []
label = ''  # outcome
filename = 'housevotes84.csv'
with open(filename) as f:
    for line_num, line in enumerate(f.readlines()):
        words = line.strip().split(",")
        add_to_NONMISSING = '?' not in words
        for ix, w in enumerate(words):
            if line_num == 0:
                data[w] = []
                NONMISSING[w] = []
                index_to_feature.append(w)
                label = w  # label is last column
            else:
                data[index_to_feature[ix]].append(w)
                if add_to_NONMISSING:
                    NONMISSING[index_to_feature[ix]].append(w)


train_all, test = split_train_test(NONMISSING)
NUM_TRIALS = 30

# build learning curve
sizes = []
accuracy_scores = []
for SIZE in range(5, len(train_all[label]) + 1):
    acc_sum = 0
    for trial in range(NUM_TRIALS):
        train, _ = split_train_test(train_all, test_set_size=(len(train_all[label]) - SIZE))
        del train[index_to_feature[0]]
        pr_tree = []
        acc_tree = make_decision_tree(train, {})
        acc_score = accuracy(test, acc_tree)
        acc_sum += acc_score
    print(SIZE, ": ", acc_sum / NUM_TRIALS)
    sizes.append(SIZE)
    accuracy_scores.append(acc_sum / NUM_TRIALS)


# plot learning curve
plt.plot(sizes, accuracy_scores, 'ro')
plt.axis([0, len(train_all[label]) + 5, 30, 105])
plt.xlabel('SIZE')
plt.ylabel('Accuracy')
plt.show()
