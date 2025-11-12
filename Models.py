import numpy as np
import time

class GradientDescent:
    def __init__(self, learning_rate=.1, max_iters=1e4, epsilon=1e-8, record_history=False, momentum=0, mode="REG"):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.record_history = record_history
        self.w_history = []
        self.gradients_history = []
        self.momentum = momentum
        self.adagrad_epsilon = 1e-8
        self.adagrad_accum = None
        self.mode = mode  # REG or ADA

    # Full batch convergence
    def run(self, gradient_fn, x, y, w, reg_factor=None):
        grad = np.inf
        t = 1
        while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:
            grad = gradient_fn(x,y,w)
            w = w - self.learning_rate * grad
            if reg_factor is not None:
                thresh = self.learning_rate * reg_factor
                w[:-1] = np.sign(w[:-1]) * np.maximum(0, np.abs(w[:-1]) - thresh)
            t += 1
        return w


    # Stochastic update for one batch
    def run_stochastic(self, gradient_fn, x, y, w):
        if self.mode == "REG":
            return self.run_stochastic_REG(gradient_fn, x, y, w)
        elif self.mode == "ADA":
            return self.run_stochastic_adagrad(gradient_fn, x, y, w)
        else:
            raise ValueError("Unknown mode for stochastic gradient descent")

    def run_stochastic_REG(self, gradient_fn, x, y, w):
        if len(self.gradients_history) == 0:
            self.gradients_history.append(np.zeros_like(w))
            self.w_history.append(w)

        grad = self.momentum * self.gradients_history[-1] + (1-self.momentum)*(gradient_fn(x, y, w))
        w = w - self.learning_rate * grad

        self.gradients_history.append(grad)
        self.w_history.append(w)
        return w


    def run_stochastic_adagrad(self, gradient_fn, x, y, w):
        if self.adagrad_accum is None:
            self.adagrad_accum = np.zeros_like(w)

        grad = gradient_fn(x, y, w)
        self.adagrad_accum += grad**2
        adjusted_lr = self.learning_rate / (np.sqrt(self.adagrad_accum + self.adagrad_epsilon))
        w = w - adjusted_lr * grad

        return w

class LinearRegression:
    def __init__(self, add_bias=True, batch_size=None, optimizer=GradientDescent(), num_epochs=10,
                 regularization_mode="None", regularization_factor=0.01):
        self.add_bias = add_bias
        self.w = None
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.time_to_fit = 0
        self.regularization_mode = regularization_mode
        self.regularization_factor = regularization_factor

    def compute_regularization(self,w):
        reg = np.zeros_like(w)
        if self.regularization_mode == "L2":
            reg[:-1] = self.regularization_factor * w[:-1]
        elif self.regularization_mode == "L1":
            reg[:-1] = self.regularization_factor * np.where(w[:-1] == 0, 0, np.sign(w[:-1]))
        return reg

    def gradient(self,x,y,w):
        yh = x @ w
        N, D = x.shape
        grad = np.dot(yh - y, x)/N + self.compute_regularization(w)
        return grad

    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]
        N,D = x.shape
        if self.add_bias:
            x = np.column_stack([x, np.ones(N)])
        
        N,D = x.shape
        self.w = np.zeros(D)
        if self.batch_size is None:
            self.batch_size = N

        # If no optimizer given, use least squares on entire dataset
        if self.optimizer is None:
            start = time.time()
            if self.regularization_mode == 'L2':
                reg_matrix = self.regularization_factor * np.eye(D)
                if self.add_bias: reg_matrix[-1,-1] = 0  
                self.w = np.linalg.solve(x.T @ x + reg_matrix, x.T @ y)
            elif self.regularization_mode == 'L1':
                raise ValueError("Closed-form solution not available for L1 regularization")
            else:
                self.w = np.linalg.lstsq(x,y, rcond=None)[0]
            self.time_to_fit = time.time() - start
            return self

        """        if self.batch_size == N:
            # run full batch gradient descent
            start = time.time()
            self.w = self.optimizer.run(self.gradient, x, y, self.w, reg_factor=self.regularization_factor if self.regularization_mode=="L1" else None)
            self.time_to_fit = time.time() - start
            return self
        """

        # Mini-batch gradient descent
        start = time.time()
        for _ in range(self.num_epochs):
            # randomize data order
            perm = np.random.permutation(N)
            x_rand = x[perm]
            y_rand = y[perm]
            for i in range(0, N, self.batch_size):
                x_batch = x_rand[i:i+self.batch_size]
                y_batch = y_rand[i:i+self.batch_size]
                self.w = self.optimizer.run_stochastic(self.gradient, x_batch, y_batch, self.w)

        self.time_to_fit = time.time() - start
        return self
    
    def predict(self, x):
        if self.add_bias:
            x = np.column_stack([x, np.ones(x.shape[0])])
        return x @ self.w
    

class LogisticRegression:
    def __init__(self, add_bias=True, learning_rate =.1,batch_size=None,
                  epsilon=1e-4, max_iters=1e5, optimizer=GradientDescent(), num_epochs=10,
                  regularization_mode="None", regularization_factor=0.01):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.logistic = lambda z: 1./ (1 + np.exp(-z))
        self.w = None
        self.time_to_fit = 0
        self.regularization_mode = regularization_mode
        self.regularization_factor = regularization_factor

    def compute_regularization(self):
        reg = np.zeros_like(self.w)
        if self.regularization_mode == "L2":
            reg[1:] = self.regularization_factor * self.w[1:]
        elif self.regularization_mode == "L1":
            reg[1:] = self.regularization_factor * np.sign(self.w[1:])
        return reg

    def gradient(self, x, y, w):
        N,D = x.shape
        yh = self.logistic(np.dot(x, w))
        grad = np.dot(x.T, yh-y)/N + self.compute_regularization()
        return grad
    
    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]
        
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x, np.ones(N)])
        
        N,D = x.shape
        if self.batch_size is None:
            self.batch_size = N

        self.w = np.zeros(D)
        
        # Full batch gradient descent
        if self.optimizer is None:
            start = time.time()
            g = np.inf
            t = 0
            self.optimizer = GradientDescent(learning_rate=self.learning_rate, epsilon=self.epsilon, max_iters=self.max_iters)
            self.w = self.optimizer.run(self.gradient, x, y, self.w)
            self.time_to_fit = time.time() - start
            return self

        # Mini-batch gradient descent
        start = time.time()
        for _ in range(self.num_epochs):
            # randomize data order
            perm = np.random.permutation(N)
            x_rand = x[perm]
            y_rand = y[perm]
            for i in range(0,N,self.batch_size):
                x_batch = x_rand[i:i+self.batch_size]
                y_batch = y_rand[i:i+self.batch_size]
                self.w = self.optimizer.run_stochastic(self.gradient, x_batch, y_batch, self.w)
        self.time_to_fit = time.time() - start
        return self
    
    def predict(self, x):
        if x.ndim == 1:
            x = x[:, None]
        Nt = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x, np.ones(Nt)])
        yh = self.logistic(np.dot(x, self.w))
        return yh
        
