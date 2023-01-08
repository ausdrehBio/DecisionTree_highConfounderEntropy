## Decision Tree, maintaining a high entropy for confounding variables
'''
Upcoming:
[ ] Perfomance Comparison

[ ] check if Confounder is categorical or continous
    Combination of categorical and continuous Confounders

[ ] what if Confounder and/or Target are not numerical?

[ ] Option to transform continous variables into categorical variables
    --> binning

'''

    

#################### custom Node class ####################

# Data wrangling 
import pandas as pd 
from math import log

# Array math
import numpy as np 

# Quick value count calculator
from collections import Counter         

class Node: 
    """
    Class for creating the nodes for a decision tree 
    """
    def __init__(   
        self, 
        Y: list,
        X: pd.DataFrame,            

        ConfounderFrame: pd.DataFrame,
        categorical: bool,

        min_samples_split=None,
        max_depth=None,
        depth=None,
        node_type=None,
        rule=None
    ):
        # Saving the data to the node 
        self.Y = Y 
        self.X = X
        self.ConfounderFrame = ConfounderFrame
        self.categorical = categorical

        # Saving the hyper parameters
        self.min_samples_split = min_samples_split if min_samples_split else 20
        self.max_depth = max_depth if max_depth else 5                          # parameter for stopping the tree from growing too deep

        # Default current depth of node 
        self.depth = depth if depth else 0      

        # Extracting all the features
        self.features = list(self.X.columns)        

        # Type of node 
        self.node_type = node_type if node_type else 'root'         
        
        # Rule for spliting 
        self.rule = rule if rule else ""    # saves the rule for splitting the node as a string

        # Calculating the counts of Y in the node 
        self.counts = Counter(Y)            

        #self.ConfounderCounts = Counter(Confounder)
        self.ConfoundingVariables = list(self.ConfounderFrame.columns)      

        self.entropy_value = self.get_TargetEntropy()


        # Sorting the counts and saving the final prediction of the node 
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))         
        

        
        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]
        

        # Saving to object attribute. This node will predict the class with the most frequent class
        self.yhat = yhat 
        
        # Saving the number of observations in the node 
        self.n = len(Y)

        
        self.left = None 
        self.right = None 

        self.best_feature = None 
        self.best_value = None 




############################## Methods
################# Entropy
    @staticmethod
    def compute_ConfounderEntropy(self,frame=None) -> float:
        if frame is None:
            frame = self.ConfounderFrame
        Entropy = 0
        # iterate over columns
        for col in frame.columns:
            # subset each column
            subset = frame[col]
            # all unique values in the column
            classes = subset.unique()

            # iterate over unique values
            for c in classes:           
                # count the number of times each unique value appears
                count = subset.value_counts()[c]    
                # calculate the probability of each unique value
                prob = count / len(subset)
                
                Entropy += (-1)*prob * np.log2(prob)
                
        return Entropy

    @staticmethod
    def compute_continuousConfounder(self,frame=None) -> float:
        if frame is None:
            frame = self.ConfounderFrame
        Randomness = 0
        # iterate over columns
        for col in frame.columns:
            # subset each column
            subset = frame[col]
            # get std of subset
            std = subset.std()
            Randomness += std
        return Randomness
                
    def get_ConfounderEntropy(self, feature, value, Xdf):
        
        left = Xdf[Xdf[feature]<value][self.ConfoundingVariables]     
        right = Xdf[Xdf[feature]>value][self.ConfoundingVariables]
        if self.categorical==True:
            ConfEntropy_left = self.compute_ConfounderEntropy(self, left)
            ConfEntropy_right = self.compute_ConfounderEntropy(self, right)
        else:
            ConfEntropy_left = self.compute_continuousConfounder(self, left)
            ConfEntropy_right = self.compute_continuousConfounder(self, right)

        return ConfEntropy_left, ConfEntropy_right


    @staticmethod
    def compute_TargetEntropy(countsDictionary) -> float: 

        # sum values of countsDictionary
        n = sum(countsDictionary.values())
        Entropy = 0
        for key in countsDictionary:
            p = countsDictionary[key]/n
            if p > 0:
                x = -p*log(p,2)
            else:
                x = 0
            Entropy += x

        return Entropy

    def get_TargetEntropy(self):     
       
        return self.compute_TargetEntropy(self.counts)

    @staticmethod
    def get_usability(Gain_of_Information, weighted_ConfounderEntropy):
        '''
        Combination of low TargetEntropy (=high gain of Information) and high ConfounderEntropy (or std of Confounder)
        '''
        #usability = (1/weighted_TargetEntropy) + weighted_ConfounderEntropy
        usability = Gain_of_Information + weighted_ConfounderEntropy

        return usability



################# Spliting
    @staticmethod
    def ma(x: np.array, window: int) -> np.array:
        
        return np.convolve(x, np.ones(window), 'valid') / window

    def best_split(self) -> tuple:
        """
        Given the X features and Y targets calculates the best split 
        for a decision tree
        """
        # Creating a dataset for spliting
        df = self.X.copy()
        df['Y'] = self.Y
        df[self.ConfoundingVariables] = self.ConfounderFrame

        # Getting the entropy for the base input 
        entropy_base = self.get_TargetEntropy()

        # Finding which split yields the best Usability 
        max_usability = 0

        
        best_feature = None
        best_value = None

        for feature in self.features:
            # Droping missing values
            Xdf = df.dropna().sort_values(feature)

            # Sorting the values and getting the rolling average
            xmeans = self.ma(Xdf[feature].unique(), 2)

            for value in xmeans:

                '''------------------Target------------------'''
                # Spliting the dataset, counting for Target-Variable 
                left_counts = Counter(Xdf[Xdf[feature]<value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature]>=value]['Y'])


                # Getting the left and right entropies for Target-Variable
                entropy_left = self.compute_TargetEntropy(left_counts)
                entropy_right = self.compute_TargetEntropy(right_counts)




                '''------------------Confounder------------------'''

                Confentropy_left, Confentropy_right = self.get_ConfounderEntropy(feature, value, Xdf)



                '''------------------for both------------------'''
                # Getting the obs count from the left and the right data splits
                n_left = sum(left_counts.values())
                n_right = sum(right_counts.values())
                # wiviele Personen in dem Node sind

                # Calculating the weights for each of the nodes
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)

                
                
                '''------------------Target------------------'''
                # Calculating the weighted Entropy for Target
                wEntropy = w_left * entropy_left + w_right * entropy_right

                '''------------------Confounder------------------'''
                # Calculating the weighted Entropy for Confounder
                wConfEntropy = w_left * Confentropy_left + w_right * Confentropy_right

                # Calculating the informationGain
                informationGain = entropy_base - wEntropy

                usability = self.get_usability(informationGain,wConfEntropy)

                # Checking if this is the best split so far 
                if usability > max_usability:
                    best_feature = feature
                    best_value = value 

                    max_usability = usability
        if self.categorical==True:
            print(f'Confounder Entropy: {wConfEntropy}')
        else:
            print(f'Confounder standard deviation: {wConfEntropy}')
        print(f'Target Entropy: {wEntropy}')

        return (best_feature, best_value)




################# Tree
    def grow_tree(self):
        """
        Recursive method to create the decision tree
        """
        # Making a df from the data 
        df = self.X.copy()
        df['Y'] = self.Y
        df[self.ConfoundingVariables] = self.ConfounderFrame

        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):

            # Getting the best split 
            best_feature, best_value = self.best_split()

            if best_feature is not None:
                
                self.best_feature = best_feature
                self.best_value = best_value

                # Getting the left and right nodes
                left_df, right_df = df[df[best_feature]<=best_value].copy(), df[df[best_feature]>best_value].copy()

                # Creating the left and right nodes
                left = Node(
                    left_df['Y'].values.tolist(), 
                    left_df[self.features],
                    left_df[self.ConfoundingVariables],
                    categorical=self.categorical, 
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split, 
                    node_type='left_node',
                    rule=f"{best_feature} <= {round(best_value, 3)}"
                    )

                self.left = left 
                self.left.grow_tree()

                right = Node(
                    right_df['Y'].values.tolist(), 
                    right_df[self.features],
                    right_df[self.ConfoundingVariables],
                    categorical=self.categorical, 
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split,
                    node_type='right_node',
                    rule=f"{best_feature} > {round(best_value, 3)}"
                    )

                self.right = right
                self.right.grow_tree()

    def print_info(self, width=4):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces 
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const
        
        if self.node_type == 'root':
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | Entropy of the node: {round(self.entropy_value, 2)}")
        print(f"{' ' * const}   | Class distribution in the node: {dict(self.counts)}")
        print(f"{' ' * const}   | Predicted class: {self.yhat}")   

    def print_tree(self):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info() 
        
        if self.left is not None: 
            self.left.print_tree()
        
        if self.right is not None:
            self.right.print_tree()





################# Predictions
    def predict(self, X:pd.DataFrame):
        """
        Batch prediction method
        """
        predictions = []

        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})
        
            predictions.append(self.predict_obs(values))
        
        return predictions

    def predict_obs(self, values: dict) -> int:
        """
        Method to predict the class given a set of features
        """
        cur_node = self
        while cur_node.depth < cur_node.max_depth:
            # Traversing the nodes all the way to the bottom
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value

            if cur_node.n < cur_node.min_samples_split:
                break 

            if (values.get(best_feature) < best_value):
                if self.left is not None:
                    cur_node = cur_node.left
            else:
                if self.right is not None:
                    cur_node = cur_node.right
            
        return cur_node.yhat
############################################################






# Loading data
d = pd.read_csv('/Users/ausdrehbem/Desktop/uni/cosybio_seminar/data_samples/MultipleContinousConfounder.csv')
d.head()
# Dropping missing values
dtree = d[['Pclass', 'Age', 'Fare', 'confounder','random','continous1','continous2']].dropna().copy()
dtree.head()
# Defining the X and Y matrices
Y = dtree['Pclass'].values
X = dtree[['Age', 'Fare']]

Conf = dtree[['continous1','continous2']]




# Saving the feature list 
features = list(X.columns)




# define the dictionary of hyperparameters
hp = {
 'max_depth': 3,
 'min_samples_split': 50
}


# initialize the root node
root = Node(Y, X, Conf,categorical=False,**hp)

# grow the tree
root.grow_tree()

# print the tree
root.print_tree()

# predict the class of the first observation
root.predict_obs(X.iloc[0].to_dict())

# predict the class of the whole dataset
root.predict(X)
