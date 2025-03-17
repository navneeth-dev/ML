import pandas as pd
import math

# Sample tennis dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast',
                'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                   'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True',
             'False', 'False', 'False', 'True', 'True', 'False', 'True'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
            'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Calculate entropy
def entropy(target_col):
    elements, counts = pd.unique(target_col), target_col.value_counts()
    entropy_val = 0
    for i in range(len(elements)):
        prob = counts[elements[i]] / len(target_col)
        entropy_val -= prob * math.log2(prob)
    return entropy_val

# Calculate information gain
def info_gain(data, split_attribute, target_name="Play"):
    total_entropy = entropy(data[target_name])
    vals, counts = pd.unique(data[split_attribute]), data[split_attribute].value_counts()
   
    weighted_entropy = 0
    for i in range(len(vals)):
        subset = data[data[split_attribute] == vals[i]][target_name]
        weighted_entropy += (counts[vals[i]] / len(data)) * entropy(subset)
   
    return total_entropy - weighted_entropy

# ID3 algorithm implementation
def id3(data, original_data, features, target_attribute="Play", parent_node_class=None):
    # If all examples are positive, return leaf node with "Yes"
    if len(pd.unique(data[target_attribute])) == 1:
        return pd.unique(data[target_attribute])[0]
   
    # If dataset is empty, return parent node's majority class
    if len(data) == 0:
        return pd.unique(original_data[target_attribute])[
            pd.value_counts(original_data[target_attribute]).argmax()]
   
    # If no features left, return majority class
    if len(features) == 0:
        return parent_node_class
   
    # Build the tree
    parent_node_class = pd.unique(data[target_attribute])[
        pd.value_counts(data[target_attribute]).argmax()]
   
    # Calculate information gain for each feature
    info_gains = {feature: info_gain(data, feature, target_attribute)
                 for feature in features}
   
    # Get best feature to split on
    best_feature = max(info_gains.items(), key=lambda x: x[1])[0]
   
    # Create tree structure
    tree = {best_feature: {}}
   
    # Remove best feature from features list
    features = [f for f in features if f != best_feature]
   
    # Grow tree for each value of best feature
    for value in pd.unique(data[best_feature]):
        sub_data = data[data[best_feature] == value]
        subtree = id3(sub_data, original_data, features, target_attribute,
                     parent_node_class)
        tree[best_feature][value] = subtree
   
    return tree

# Main execution
def main():
    features = list(df.columns[:-1])  # All columns except 'Play'
    decision_tree = id3(df, df, features)
   
    # Pretty print the tree
    def print_tree(tree, level=0):
        if not isinstance(tree, dict):
            print("\t" * level + f"Leaf: {tree}")
            return
        for attribute, branches in tree.items():
            print("\t" * level + f"{attribute}")
            for value, subtree in branches.items():
                print("\t" * (level + 1) + f"{value} ->")
                print_tree(subtree, level + 2)
   
    print("Decision Tree:")
    print_tree(decision_tree)

if __name__ == "__main__":
    main()
