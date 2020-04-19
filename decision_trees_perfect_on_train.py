import math


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


def make_decision_tree(dataset, depth):
    best_feature, info_gain = choose_feature(dataset)
    tree.append(("%s? (information gain: %s)" % (best_feature, info_gain), depth))
    for val in set(dataset[best_feature]):
        red_dataset = reduce_dataset(best_feature, val, dataset)
        red_ent = entropy(red_dataset[label])
        if red_ent == 0:
            tree.append(("%s --> %s" % (val, red_dataset[label][0]), depth + 1))
        else:
            tree.append(("%s (with current entropy %s)" % (val, red_ent), depth + 1))
            make_decision_tree(red_dataset, depth + 2)


def print_tree(dec_tree):
    print(" * Starting Entropy:", entropy(data[label]))
    for element, depth in dec_tree:
        print("  " * depth, "*", element)


data = {}
index_to_feature = []
label = ''
filename = 'mushroom.csv'
with open(filename) as f:
    for line_num, line in enumerate(f.readlines()):
        words = line.strip().split(",")
        for ix, w in enumerate(words):
            if line_num == 0:
                data[w] = []
                index_to_feature.append(w)
                label = w  # label is last column
            else:
                data[index_to_feature[ix]].append(w)

tree = []
make_decision_tree(data, 0)
print_tree(tree)
