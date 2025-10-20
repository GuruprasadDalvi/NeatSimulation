import random
import json
import numpy as np

# Optional GPU backend (CuPy). Falls back to NumPy if unavailable
try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False
def _sigmoid(x, xp):
    return 1 / (1 + xp.exp(-x))

def _d_sigmoid(x, xp):
    s = _sigmoid(x, xp)
    return s * (1 - s)

def _tanh(x, xp):
    return xp.tanh(x)

def _d_tanh(x, xp):
    y = xp.tanh(x)
    return 1 - (y * y)

def cost_function(y_true, y_pred, xp=np):
    y_true = y_true.reshape((len(y_true), 1))
    diff = (y_true - y_pred)
    cost = xp.sum((diff ** 2) ** 2)
    return cost

class nNetwork:
    def __init__(self, layers, activeFun="sigmoid", device: str = "auto"):
        '''
            Layers: list of number of layes including input layer and output layer
            activeFun(Activation Function) : sigmoid(defalut), tanh
            device: 'auto' (prefer GPU if available), 'gpu', or 'cpu'
        '''
        self.layers = layers
        self.activeFun = activeFun
        if device not in ("auto", "gpu", "cpu"):
            device = "auto"
        use_gpu = (_HAS_CUPY and device in ("auto", "gpu"))
        self.xp = cp if use_gpu else np

        # Internal XP weights/biases for compute; public NumPy copies for UI/saving
        self._weights = []
        for i in range(len(layers)):
            if i != 0:
                h = self.layers[i-1]
                w = self.layers[i]
                we = self.xp.random.randn(w, h)
                self._weights.append(we)
        self._baise = [self.xp.random.rand(n, 1) for n in self.layers[1:]]

        # Public CPU-visible copies
        self.weights = [self._to_cpu(w) for w in self._weights]
        self.baise = [self._to_cpu(b) for b in self._baise]

        # caches for backprop
        self._z = []
        self.z = []
        self._activisions = []
        self.activisions = []

    def _to_cpu(self, arr):
        if _HAS_CUPY and isinstance(arr, cp.ndarray):  # type: ignore
            return cp.asnumpy(arr)
        return arr

    def _to_xp(self, arr):
        return self.xp.array(arr)

    def __deepcopy__(self, memo):
        # Preserve device selection while avoiding deepcopy of module objects
        device = "gpu" if (_HAS_CUPY and self.xp is cp) else "cpu"
        cloned = nNetwork(self.layers[:], activeFun=self.activeFun, device=device)
        # Copy XP tensors
        cloned._weights = [cloned.xp.array(w) for w in self._weights]
        cloned._baise = [cloned.xp.array(b) for b in self._baise]
        # Mirror CPU-visible copies
        cloned.weights = [cloned._to_cpu(w) for w in cloned._weights]
        cloned.baise = [cloned._to_cpu(b) for b in cloned._baise]
        # Clear caches
        cloned._z = []
        cloned.z = []
        cloned._activisions = []
        cloned.activisions = []
        return cloned



    def activate(self, x, deriv=False):
        if deriv:
            if self.activeFun == "sigmoid":
                return _d_sigmoid(x, self.xp)
            if self.activeFun == "tanh":
                return _d_tanh(x, self.xp)
        if self.activeFun == "sigmoid":
            return _sigmoid(x, self.xp)
        if self.activeFun == "tanh":
            return _tanh(x, self.xp)

    def feedforward(self, inputs):
        xp = self.xp
        a = xp.array(inputs).reshape((len(inputs), 1))
        self._activisions = [a]
        self.activisions = [self._to_cpu(a)]
        self._z = []
        self.z = []
        for i in range(len(self.layers) - 1):
            w = self._weights[i]
            b = self._baise[i]
            z = xp.dot(w, a) + b
            a = self.activate(z)
            self._z.append(z)
            self.z.append(self._to_cpu(z))
            self._activisions.append(a)
            self.activisions.append(self._to_cpu(a))
        # Return CPU output for external logic
        return self.activisions[-1]

    def back_prop(self, expected):
        xp = self.xp
        expected = xp.array(expected).reshape((len(expected), 1))

        # calculate all deltas using XP tensors
        err = expected - self._activisions[-1]
        delta_l = err * self.activate(self._z[-1], deriv=True)
        deltas = [0] * (len(self.layers) - 1)
        deltas[-1] = delta_l
        for i in range(len(deltas) - 2, -1, -1):
            delta = xp.dot(self._weights[i+1].transpose(), deltas[i+1]) * self.activate(self._z[i], deriv=True)
            deltas[i] = delta.reshape((len(delta), 1))

        # change weights
        dw = []
        db = []
        deltas = [0] + deltas

        for i in range(1, len(self.layers)):
            t = self._activisions[i-1].transpose()
            dw_temp = xp.dot(deltas[i], t)
            db_temp = deltas[i]
            dw.append(dw_temp)
            db.append(db_temp)

        for i in range(len(self._weights)):
            self._weights[i] += dw[i]
        for i in range(len(self._baise)):
            self._baise[i] += db[i]

        # Mirror updated XP tensors to CPU-visible copies
        self.weights = [self._to_cpu(w) for w in self._weights]
        self.baise = [self._to_cpu(b) for b in self._baise]

    def mutate(self):
        xp = self.xp
        for i in range(len(self.layers) - 1):
            # Bias mutation
            self._baise[i] += xp.random.uniform(low=-0.09999, high=0.09999, size=self._baise[i].shape)
            # Weight mutation
            self._weights[i] += xp.random.uniform(low=-1.0, high=1.0, size=self._weights[i].shape)
        # Mirror to CPU copies
        self.weights = [self._to_cpu(w) for w in self._weights]
        self.baise = [self._to_cpu(b) for b in self._baise]
       


    def predict(self, inputs):
        xp = self.xp
        a = xp.array(inputs).reshape((len(inputs), 1))
        activisions_xp = [a]
        activisions_cpu = [self._to_cpu(a)]
        for i in range(len(self.layers) - 1):
            w = self._weights[i]
            b = self._baise[i]
            d = xp.dot(w, a)
            a = self.activate(d + b)
            activisions_xp.append(a)
            activisions_cpu.append(self._to_cpu(a))
        # expose CPU activations for visualization
        self._activisions = activisions_xp
        self.activisions = activisions_cpu
        return activisions_cpu[-1]
    
    def save(self, filename="network.json"):
        we = {}
        counter = 0
        for wei in self.weights:
            we[counter] = wei.tolist()
            counter += 1
        bai = {}
        counter = 0
        for b in self.baise:
            bai[counter] = b.tolist()
            counter += 1
        networkDic = {
            "layers": self.layers,
            "weights": we,
            "baises": bai
        }
        op_file = open(filename, "w")
        json.dump(networkDic, op_file, indent=3)
        op_file.close()

    def load(self, path):
        # loading already saved network
        with open(path, 'r') as target:
            data = json.load(target)
        xp = self.xp
        ls = data['layers']
        ws_cpu = []
        bs_cpu = []
        for w in data['weights']:
            l = data['weights'][w]
            ws_cpu.append(np.array(l))
        for b in data['baises']:
            l = data['baises'][b]
            bs_cpu.append(np.array(l))

        # set CPU-visible copies
        self.layers = ls
        self.weights = ws_cpu
        self.baise = bs_cpu
        # create XP tensors for compute
        self._weights = [xp.array(w) for w in ws_cpu]
        self._baise = [xp.array(b) for b in bs_cpu]
        print('Network Loaded Succsefully')
        
    def crossover(self,partner):
        child=nNetwork(self.layers,activeFun=self.activeFun)
        for i in range(len(self.layers)-1):
            if i%2==0:
                child.weights[i]=self.weights[i]
            else:
                child.weights[i]=partner.weights[i]
        for i in range(len(self.layers)-1):
            if i%2==0:
                child.baise[i] = self.baise[i]
            else:
                child.baise[i] = partner.baise[i]
        return child




# inputs=np.array([50,28,95,69,10],dtype=np.float)
# expected_outputs=np.array([-13,50,-26,34],dtype=np.float)
# inputs=inputs/1000
# expected_outputs=expected_outputs/1000


# net=nNetwork([5,16,4],activeFun="tanh")
# net.feedforward(inputs)
# net.back_prop(expected_outputs)
# c=cost_function(expected_outputs,net.activisions[-1])
# print(c)
# for i in range(350):
#     net.feedforward(inputs)
#     net.back_prop(expected_outputs)
#     c=cost_function(expected_outputs,net.activisions[-1])
#     #print(c)

# net.predict(inputs)
# print(c)
    
