from elman_neural_network.neural_network import MLPClassifier


mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', batch_size=5)


from sklearn.datasets import load_iris


iris = load_iris()

print('here')

mlp.fit(iris['data'], iris['target'])

y_hat = mlp.predict(iris['data'])

from sklearn.metrics import confusion_matrix


print(confusion_matrix(iris['target'], y_hat))