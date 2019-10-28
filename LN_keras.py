import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import get as get_activation
from tensorflow.keras.layers import SimpleRNNCell, RNN, Layer
from tensorflow.keras.layers.experimental import LayerNormalization

class SimpleRNNCellWithLayerNorm(SimpleRNNCell):
    def __init__(self, units, **kwargs):
        self.activation = get_activation(kwargs.get("activation", "tanh"))
        kwargs["activation"] = None
        super().__init__(units, **kwargs)
        self.layer_norm = LayerNormalization()
    def call(self, inputs, states):
        outputs, new_states = super().call(inputs, states)
        norm_out = self.activation(self.layer_norm(outputs))
        return norm_out, [norm_out]
		
model = Sequential([
    RNN(SimpleRNNCellWithLayerNorm(20), return_sequences=True,
        input_shape=[None, 20]),
    RNN(SimpleRNNCellWithLayerNorm(5)),
])

model.compile(loss="mse", optimizer="sgd")
X_train = np.random.randn(100, 50, 20)
Y_train = np.random.randn(100, 5)
history = model.fit(X_train, Y_train, epochs=2)