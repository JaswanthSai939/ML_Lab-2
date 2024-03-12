#1st question
import math

def Euclidean_Distance(Vector1, Vector2):
    if len(Vector1) != len(Vector2):
        raise ValueError("Vectors must have the same dimensions")
    
    distance = 0
    for i in range(len(Vector1)):
        distance += (Vector1[i] - Vector2[i]) ** 2
    
    return math.sqrt(distance)

def Manhattan_Distance(Vector1, Vector2):
    if len(Vector1) != len(Vector2):
        raise ValueError("Vectors must have the same dimensions")
    
    distance = 0
    for i in range(len(Vector1)):
        distance += abs(Vector1[i] - Vector2[i])
    
    return distance

# Example usage:
Vector_a = [2, 8, 9]
Vector_b = [1, 4, 3]

euclidean_dist = Euclidean_Distance(Vector_a, Vector_b)
manhattan_dist = Manhattan_Distance(Vector_a, Vector_b)

print(f"Euclidean Distance: {euclidean_dist}")
print(f"Manhattan Distance: {manhattan_dist}")

#2nd question
from sklearn.neighbors import KNeighborsClassifier

def knn_classifier(X_train, y_train, X_test, k):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  return y_pred

# Example usage:
X_train = [[1, 2], [3, 4], [5, 6], [7, 8]]
y_train = [0, 0, 1, 1]
X_test = [[1, 1], [5, 5]]
k = 3

y_pred = knn_classifier(X_train, y_train, X_test, k)

print(y_pred)


#3rd question
def label_encoding(cateogries):
    unique_cateogries=set(cateogries)
    label_map={}

    for i, cateogries in enumerate(unique_cateogries):
        label_map[cateogries]=i

    return label_map
cateogries=['cat', 'dog', 'cat', 'bird', 'dog', 'cat']
label_map=label_encoding(cateogries)
print("label encoding:",label_map)

#4th question
def one_hot_encoding(cateogries):
    unique_cateogries=sorted(set(cateogries))
    encoding=[]
    for cateogry in cateogries:
        one_hot_vector=[0]*len(unique_cateogries)
        index=unique_cateogries.index(cateogry)
        one_hot_vector[index]=1
        encoding.append(one_hot_vector)

    return encoding
categories = ['red', 'blue', 'green', 'red', 'green', 'blue']

one_hot_encoded = one_hot_encoding(categories)
print("One-Hot Encoded:")
for category, one_hot_vector in zip(categories, one_hot_encoded):
    print(category, "->", one_hot_vector)
