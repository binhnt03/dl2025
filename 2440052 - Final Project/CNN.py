import matrix as mat 
from LCG import LinearCongruentialGenerator as lcg
import numpy as np

class CNN:
    def __init__(self, num_filters=2, filter_size=3, input_channels=1, num_classes=10):
        # Initialize filters for convolutional layer (num_filters, input_channels, filter_size, filter_size)
        self.filters = np.random.randn(num_filters, input_channels, filter_size, filter_size) * 0.01
        self.filter_size = filter_size
        # Initialize weights for fully connected layer
        # After conv and pooling, assume input is 13x13 (for 28x28 input after one 2x2 pooling)
        self.fc_weights = np.random.randn(num_filters * 13 * 13, num_classes) * 0.01
        self.fc_bias = np.zeros((1, num_classes))
        
    def conv_forward(self, x):
        # Input x: (height, width, channels)
        # Output: (height - filter_size + 1, width - filter_size + 1, num_filters)
        h, w, c = x.shape
        f = self.filter_size
        n_f = self.filters.shape[0]
        out_h = h - f + 1
        out_w = w - f + 1
        output = np.zeros((out_h, out_w, n_f))
        
        for i in range(out_h):
            for j in range(out_w):
                for k in range(n_f):
                    patch = x[i:i+f, j:j+f, :]
                    output[i, j, k] = np.sum(patch * self.filters[k])
        return output
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def max_pool(self, x, pool_size=2, stride=2):
        # Input x: (height, width, channels)
        # Output: (height//stride, width//stride, channels)
        h, w, c = x.shape
        out_h = h // stride
        out_w = w // stride
        output = np.zeros((out_h, out_w, c))
        
        for i in range(out_h):
            for j in range(out_w):
                for k in range(c):
                    patch = x[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size, k]
                    output[i, j, k] = np.max(patch)
        return output
    
    def fully_connected(self, x):
        # Input x: flattened feature map
        # Output: class scores
        return np.dot(x, self.fc_weights) + self.fc_bias
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x):
        # x: (height, width, channels)
        conv_out = self.conv_forward(x)
        relu_out = self.relu(conv_out)
        pool_out = self.max_pool(relu_out)
        flattened = pool_out.reshape(1, -1)
        fc_out = self.fully_connected(flattened)
        scores = self.softmax(fc_out)
        return scores, (conv_out, relu_out, pool_out, flattened)
    
    def compute_loss(self, scores, y):
        # y: true label (integer)
        # Cross-entropy loss
        target = np.zeros(scores.shape)
        target[0, y] = 1
        loss = -np.sum(target * np.log(scores + 1e-10))
        grad = scores - target  # Gradient of loss w.r.t. scores
        return loss, grad
    
    def backward(self, x, y, cache, grad_scores, learning_rate=0.01):
        conv_out, relu_out, pool_out, flattened = cache
        
        # Backprop through fully connected layer
        grad_flattened = np.dot(grad_scores, self.fc_weights.T)
        grad_fc_weights = np.dot(flattened.T, grad_scores)
        grad_fc_bias = grad_scores
        
        # Update fully connected weights
        self.fc_weights -= learning_rate * grad_fc_weights
        self.fc_bias -= learning_rate * grad_fc_bias
        
        # Reshape gradient for pooling layer
        grad_pool = grad_flattened.reshape(pool_out.shape)
        
        # Backprop through max pooling (approximate, assumes max index known)
        grad_relu = np.zeros_like(relu_out)
        pool_size, stride = 2, 2
        for i in range(grad_pool.shape[0]):
            for j in range(grad_pool.shape[1]):
                for k in range(grad_pool.shape[2]):
                    patch = relu_out[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size, k]
                    max_idx = np.argmax(patch)
                    max_i, max_j = np.unravel_index(max_idx, (pool_size, pool_size))
                    grad_relu[i*stride+max_i, j*stride+max_j, k] = grad_pool[i, j, k]
        
        # Backprop through ReLU
        grad_conv = grad_relu * (relu_out > 0)
        
        # Backprop through convolution
        grad_filters = np.zeros_like(self.filters)
        h, w, c = x.shape
        f = self.filter_size
        out_h, out_w = grad_conv.shape[:2]
        
        for i in range(out_h):
            for j in range(out_w):
                for k in range(self.filters.shape[0]):
                    patch = x[i:i+f, j:j+f, :]
                    grad_filters[k] += grad_conv[i, j, k] * patch
        
        # Update filters
        self.filters -= learning_rate * grad_filters
    
    def train_step(self, x, y, learning_rate=0.01):
        # x: input image (height, width, channels)
        # y: true label (integer)
        scores, cache = self.forward(x)
        loss, grad_scores = self.compute_loss(scores, y)
        self.backward(x, y, cache, grad_scores, learning_rate)
        return loss

# Example usage
if __name__ == "__main__":
    # Dummy input: 28x28 grayscale image
    x = np.random.randn(28, 28, 1)
    y = 5  # Dummy label
    cnn = CNN()
    loss = cnn.train_step(x, y)
    print(f"Loss: {loss}")


