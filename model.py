model = NeuralNetwork(learning_rate=0.1, max_iter=1, verbose=False)

#input of first layer must match the num of feature in the training data
model.add(Layer("relu", 4, 10))

model.add(Layer("relu", 10, 10))

#output of the layer must match the num of feature in the target classes
model.add(Layer("sigmoid", 10, 3))

# print(model)

model.fit(train, target)