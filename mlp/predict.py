import pickle

nn = pickle.load(open('best_model.pkl', 'rb'))

x = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776,
     0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053]

# expecting prediction [[0.0158427 0.9841573]]
print(nn.predict(x))

# TODO: binary cross entropy loss
# TODO: check requirements
