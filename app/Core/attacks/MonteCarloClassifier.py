from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
import numpy as np
from sklearn.tree import _tree

class MonteCarloDecisionTreeClassifier(DecisionTreeClassifier):
    """
    A variant of the Decision Tree Classifier incorporating Monte Carlo methods, this classifier introduces an element of 
    uncertainty into the decision-making process of the tree, thereby facilitating probabilistic classification. 
    It is capable of supporting various probability types to influence the path selection at each node during the prediction phase.

    This implementation is based on the article "TTTS: Tree Test Time Simulation for Enhancing Decision Tree Robustness Against Adversarial Examples", 
    authored by Seffe Cohen, Ofir Arbili, Yisroel Mirsky, and Lior Rokach, and published in AAAI, 2024.

    Parameters:

    prob_type (str):    Specifies the type of probability utilized for decision-making at each node. 
                        Available options are 'fixed', 'depth', 'certainty', 'agreement', 'distance', 'confidence', and 'bayes'.
    n_simulations (int): Determines the number of simulations to perform for each prediction to compute the average probabilities.

    Additional parameters are derived from the base class, DecisionTreeClassifier.

    """
    
    def __init__(self, prob_type='depth', n_simulations=10, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0):
        # Initialize the superclass with the provided parameters.
        super().__init__(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, random_state=random_state, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, class_weight=class_weight, ccp_alpha=ccp_alpha)
        # Validate and set the probability type and number of simulations.
        if prob_type not in ['fixed', 'depth', 'certainty', 'agreement', 'distance', 'confidence']:
            raise ValueError('Invalid prob_type')
        self.prob_type = prob_type
        self.n_simulations = n_simulations
   
    def get_depth_based_probability(self, depth):
        """
        Calculates the probability based on the depth of the node.
        
        Parameters:
        - depth (int): The depth of the current node in the tree.
        
        Returns:
        - Probability (float): A value between 0 and 0.2, increasing with the depth of the node.
        """
        return min(0.05 * depth, 0.2)

    def get_certainty_based_probability(self, node_id):
        """
        Calculates the probability based on the certainty of the prediction at the current node.
        
        Parameters:
        - node_id (int): The ID of the current node in the tree.
        
        Returns:
        - Probability (float): A value representing uncertainty, based on the distribution of class samples at the node.
        """
        node_values = self.tree_.value[node_id].flatten()
        total = np.sum(node_values)
        distribution = node_values / total
        max_certainty = np.max(distribution)
        p = 1 - max_certainty
        return min(p, 0.5)
    
    def get_agreement_based_probability(self, node_id):
        """
        Calculates the probability based on the agreement or majority class ratio at the current node.
        
        Parameters:
        - node_id (int): The ID of the current node in the tree.
        
        Returns:
        - Probability (float): A value representing uncertainty, based on the ratio of the majority class to total samples at the node.
        """
        node_values = self.tree_.value[node_id].flatten()
        majority_class_ratio = np.max(node_values) / np.sum(node_values)
        p = 1 - majority_class_ratio
        return min(p, 0.5)
    
    def get_distance_based_probability(self, node_id, sample):
        """
        Calculates the probability based on the distance of the sample's feature value from the threshold at the current node.
        
        Parameters:
        - node_id (int): The ID of the current node in the tree.
        - sample (array-like): The input sample being classified.
        
        Returns:
        - Probability (float): A value representing uncertainty, based on the normalized distance of the sample's feature value from the node's threshold.
        """
        feature_index = self.tree_.feature[node_id]
        distance = abs(sample[feature_index] - self.tree_.threshold[node_id])
        max_distance = np.max(abs(self.tree_.threshold))
        p = 0.1-min((distance/max_distance),0.1)
        return min(p, 0.5)

    def get_confidence_based_probability(self, X, node_id, sample):
        """
        Calculates the probability based on the confidence of the feature value at the current node.
        
        Parameters:
        - X (array-like): The input samples array.
        - node_id (int): The ID of the current node in the tree.
        - sample (array-like): The input sample being classified.
        
        Returns:
        - Probability (float): A value representing the confidence of the feature value, based on its distance from the mean, normalized by the standard deviation.
        """
        feature_index = self.tree_.feature[node_id]
        if feature_index == _tree.TREE_UNDEFINED:
            return 0
        
        feature_values = X[:, feature_index]
        avg = np.mean(feature_values)
        std = np.std(feature_values)
        
        distance = abs(sample[feature_index] - avg)
        p = max(0.1 - (distance / (std + 1e-9)), 0)
        return p

    
    def get_bayes_based_probability(self, node_id, sample):
        """
        Calculates the Bayesian probability at the current node.
        
        Parameters:
        - node_id (int): The ID of the current node in the tree.
        - sample (array-like): The input sample being classified.
        
        Returns:
        - Probability (float): A value representing the probability of the parent given the child, using Bayes' theorem.
        """
        if self.tree_.children_left[node_id] == _tree.TREE_LEAF or self.tree_.children_right[node_id] == _tree.TREE_LEAF:
            # Can't calculate Bayesian probability at the leaf node.
            return 0
        
        parent_samples = self.tree_.n_node_samples[node_id]
        left_child_samples = self.tree_.n_node_samples[self.tree_.children_left[node_id]]
        right_child_samples = self.tree_.n_node_samples[self.tree_.children_right[node_id]]
        
        # P(child)
        p_child = left_child_samples / parent_samples if sample[self.tree_.feature[node_id]] <= self.tree_.threshold[node_id] else right_child_samples / parent_samples
        # P(parent)
        p_parent = 1  # As we are already at the parent
        # P(child | parent) can be assumed to be proportional to how well-distributed the child is
        p_child_given_parent = 0.5  # for the sake of simplicity
        
        # P(parent | child) using Bayes' theorem
        p_parent_given_child = p_child_given_parent * p_parent / p_child
        return p_parent_given_child
    
    def traverse_tree(self, node, sample, X, depth=0):
        """
        Recursively traverses the tree to classify a sample, with a probabilistic decision at each node.
        
        Parameters:
        - node (int): The current node ID in the tree.
        - sample (array-like): The input sample being classified.
        - X (array-like): The input samples array.
        - depth (int): The depth of the current node in the tree.
        
        Returns:
        - The predicted class probabilities for the input sample.
        """
        # Determine the probability type and calculate the probability for the current node.
        if self.prob_type == 'fixed':
            p = 0.05
        elif self.prob_type == 'depth':
            p = self.get_depth_based_probability(depth)
        elif self.prob_type == 'certainty':
            p = self.get_certainty_based_probability(node)
        elif self.prob_type == 'agreement':
            p = self.get_agreement_based_probability(node)
        elif self.prob_type == 'distance':
            p = self.get_distance_based_probability(node, sample)
        elif self.prob_type == 'confidence':
            p = self.get_confidence_based_probability(X, node, sample)
        elif self.prob_type == 'bayes':
            p = self.get_bayes_based_probability(node, sample)
        else:
            raise ValueError('Invalid prob_type')

        # Traverse the tree based on the calculated probability and the sample's feature value.
        if self.tree_.feature[node] != _tree.TREE_UNDEFINED:
            # internal node
            if sample[self.tree_.feature[node]] <= self.tree_.threshold[node]:
                # go to the left child with high probability
                if np.random.rand() > p:
                    return self.traverse_tree(self.tree_.children_left[node], sample, X, depth + 1)
                else:
                    return self.traverse_tree(self.tree_.children_right[node], sample, X, depth + 1)
            else:
                # go to the right child with high probability
                if np.random.rand() > p:
                    return self.traverse_tree(self.tree_.children_right[node], sample, X, depth + 1)
                else:
                    return self.traverse_tree(self.tree_.children_left[node], sample, X, depth + 1)
        else:
            # leaf
            return self.tree_.value[node]

    def predict_proba(self, X, n_simulations=None):
        """
        Predict class probabilities for X, averaging over multiple simulations.
        
        Parameters:
        - X (array-like): The input samples array.
        - n_simulations (int, optional): The number of simulations to run for each sample. Defaults to the value set in the constructor.
        
        Returns:
        - proba (array-like): The predicted class probabilities for the input samples.
        """
        # Check if the number of simulations is set, otherwise use the default.
        if n_simulations is None:
            n_simulations = self.n_simulations
        
        # Check if the classifier is fitted and validate the input.
        check_is_fitted(self)
        X = super()._validate_X_predict(X, check_input=True)

        # Predict class probabilities for each sample, averaging over multiple simulations.
        proba = []
        for x in X:
            simulation_results = [self.traverse_tree(0, x, X).ravel() for _ in range(n_simulations)]
            simulation_results = [arr/arr.sum() for arr in simulation_results]
            mean_proba = np.mean(simulation_results, axis=0)
            proba.append(mean_proba)

        return np.array(proba)
    


class MonteCarloRandomForestClassifier(RandomForestClassifier):
    """
    Implementation of a Monte Carlo Random Forest Classifier.
    This implementation is based on the paper titled 'TTTS: Tree Test Time Simulation for Enhancing Decision Tree Robustness
    Against Adversarial Examples,' authored by Cohen Seffi, Arbili Ofir, Mirsky Yisroel, and Rokach Lior, and published in
    the proceedings of the AAAI 2024 conference.
    """

    def __init__(
        self,
        prob_type="fixed",
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        """
        Initializes the MonteCarloRandomForestClassifier.
        :param prob_type: The type of probability calculation method to use for determining the traversal path at each node.
        """
        super().__init__(
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None,
            monotonic_cst=None,
        )
        if prob_type not in [
            "fixed",
            "depth",
            "certainty",
            "agreement",
            "bayes",
            "confidence",
            "distance",
        ]:
            raise ValueError("Invalid prob_type")
        self.prob_type = prob_type

    # Probability computation methods based on different criteria:

    def get_depth_based_probability(self, depth):
        # Compute probability based on the depth of the node.
        return min(0.05 * depth, 0.2)

    def get_certainty_based_probability(self, node_id, tree):
        # Compute probability based on the certainty of the classification at the node.
        node_values = tree.value[node_id].flatten()
        total = np.sum(node_values)
        distribution = node_values / total
        max_certainty = np.max(distribution)
        p = 0.5 - max_certainty
        return max(p, 0)

    def get_agreement_based_probability(self, node_id, tree):
        # Compute probability based on the agreement (majority class ratio) at the node.
        node_values = tree.value[node_id].flatten()
        majority_class_ratio = np.max(node_values) / np.sum(node_values)
        p = 0.5 - majority_class_ratio
        return max(p, 0)

    def get_confidence_based_probability(self, X, node_id, sample, tree):
        # Compute probability based on the confidence (distance from mean normalized by standard deviation).
        feature_index = tree.feature[node_id]
        if feature_index == _tree.TREE_UNDEFINED:
            return 0

        feature_values = X[:, feature_index]
        avg = np.mean(feature_values)
        std = np.std(feature_values)

        distance = abs(sample[feature_index] - avg)
        p = max(0.5 - (distance / (std + 1e-9)), 0)
        return p

    def get_bayes_based_probability(self, node_id, sample, tree):
        # Compute Bayesian probability, considering the prior, likelihood, and marginal likelihood.
        if (
            tree.children_left[node_id] == _tree.TREE_LEAF
            or tree.children_right[node_id] == _tree.TREE_LEAF
        ):
            # Can't calculate Bayesian probability at the leaf node.
            return 0

        parent_values = tree.value[node_id].flatten()
        parent_samples = np.sum(parent_values)

        left_child_values = tree.value[tree.children_left[node_id]].flatten()
        right_child_values = tree.value[tree.children_right[node_id]].flatten()

        prior = np.max(parent_values) / parent_samples

        left_majority_ratio = np.max(left_child_values) / np.sum(left_child_values)
        right_majority_ratio = np.max(right_child_values) / np.sum(right_child_values)
        likelihood = left_majority_ratio * right_majority_ratio

        marginal_likelihood = np.mean([left_majority_ratio, right_majority_ratio])

        posterior = likelihood * (prior / (marginal_likelihood + 1e-9))

        p = 0.5 - posterior
        # Return the scaled posterior
        return max(p, 0)

    def get_distance_based_probability(self, X, tree, node_id, sample):
        # Compute probability based on the distance of the sample's feature value from the threshold.
        feature_index = tree.feature[node_id]
        if feature_index == _tree.TREE_UNDEFINED:
            return 0

        threshold = tree.threshold[node_id]
        feature_value = sample[feature_index]
        feature_values = X[:, feature_index]
        distance = abs(feature_value - threshold)
        std = np.std(feature_values)

        # The closer the distance is to 0, the lower the probability
        p = max(0.5 - (distance / (std + 1e-9)), 0)

        return p

    def traverse_tree(self, tree, node, sample, X, depth=0):
        # Recursively traverse the tree to make a prediction for a sample.
        if self.prob_type == "fixed":
            p = 0.05
        elif self.prob_type == "depth":
            p = self.get_depth_based_probability(depth)
        elif self.prob_type == "certainty":
            p = self.get_certainty_based_probability(node, tree)
        elif self.prob_type == "agreement":
            p = self.get_agreement_based_probability(node, tree)
        elif self.prob_type == "confidence":
            p = self.get_confidence_based_probability(X, node, sample, tree)
        elif self.prob_type == "bayes":
            p = self.get_bayes_based_probability(node, sample, tree)
        elif self.prob_type == "distance":
            p = self.get_distance_based_probability(X, tree, node, sample)
        else:
            raise ValueError("Invalid prob_type")

        # Decision to traverse left or right child node based on the computed probability.
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            if sample[tree.feature[node]] <= tree.threshold[node]:
                if np.random.rand() > p:
                    return self.traverse_tree(
                        tree, tree.children_left[node], sample, X, depth + 1
                    )
                else:
                    return self.traverse_tree(
                        tree, tree.children_right[node], sample, X, depth + 1
                    )
            else:
                if np.random.rand() > p:
                    return self.traverse_tree(
                        tree, tree.children_right[node], sample, X, depth + 1
                    )
                else:
                    return self.traverse_tree(
                        tree, tree.children_left[node], sample, X, depth + 1
                    )
        else:
            # Return the value at the leaf node.
            return tree.value[node]

    def predict_proba(self, X, n_simulations=10):
        # Make a probability prediction for each sample in X, based on n_simulations.
        check_is_fitted(self)
        X = self._validate_X_predict(X)

        proba = []
        for x in X:
            simulation_results = []
            for tree in self.estimators_:
                tree_results = [
                    self.traverse_tree(tree.tree_, 0, x, X).ravel()
                    for _ in range(n_simulations)
                ]
                simulation_results.extend(tree_results)
            mean_proba = np.mean(simulation_results, axis=0)
            proba.append(mean_proba)

        return np.array(proba)