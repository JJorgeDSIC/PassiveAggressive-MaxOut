import numpy as np

class PassiveAggressiveMaxOut:

    def __init__(self, c_0=1.0, c_1=1.0, alpha=0.9, hidden_units=32, pieces=2, a=-0.1, b=0.1, always_update=False, debug=False):

        self.C_0 = c_0
        self.C_1 = c_1
        self.alpha = alpha
        self.hidden_units = hidden_units
        self.pieces = pieces

        self.a = a
        self.b = b

        self.debug = debug

        self.epsilon = 0.0

        self.input_units = -1

        self.w_0 = None
        self.w_1 = None

        self.z_t = None
        self.max_index = None
        self.mistakes = None

        # For Mapping labels
        self.original_labels = None
        self.label_mapping = None
        self.inverse_label_mapping = None

        self.cumulative_error = 0
        self.cumulative_error_rate = 0

        self.always_update = always_update

        np.random.seed(0)

    @staticmethod
    def gs(w, row_vecs=True, norm=True):

        X = w.copy()

        if not row_vecs:
            X = X.T

        Y = X[0:1, :].copy()
        for i in range(1, X.shape[0]):
            proj = np.diag((X[i, :].dot(Y.T) / np.linalg.norm(Y, axis=1) ** 2).flat).dot(Y)
            Y = np.vstack((Y, X[i, :] - proj.sum(0)))
        if norm:
            Y = np.diag(1 / np.linalg.norm(Y, axis=1)).dot(Y)

        if row_vecs:
            return Y
        else:
            return Y.T

    @staticmethod
    def get_mod_norm(x_t):
        return np.sum(x_t * x_t)

    def init_weights(self, input_units=None, pieces=None, hidden_units=None):

        self.w_0 = np.random.uniform(self.a, self.b, (input_units + 1, pieces * hidden_units)).astype(np.float32)

        for hidden_unit in xrange(hidden_units):
            base = hidden_unit * pieces

            self.w_0[:, base:base + pieces] = self.gs(self.w_0[:, base:base + pieces], row_vecs=False, norm=True)
            
        self.w_1 = np.random.uniform(self.a, self.b, hidden_units).astype(np.float32)

    def fit_batch(self, X, Y, w_0=None, w_1=None):

        n_samples = X.shape[0]

        # Mapping labels
        self.original_labels = list(set(Y))
        self.label_mapping = dict(zip(self.original_labels, [-1, 1]))
        self.inverse_label_mapping = dict(zip([-1, 1], self.original_labels))

        y_true = np.zeros((Y.shape[0]), dtype=np.int32)

        for n_sample in xrange(n_samples):
            y_true[n_sample] = self.label_mapping[Y[n_sample]]

        self.input_units = X.shape[1]

        if w_0 is None and w_1 is None:
            self.init_weights(input_units=self.input_units, pieces=self.pieces, hidden_units=self.hidden_units)
        else:
            self.w_0 = w_0
            self.w_1 = w_1

        counter = 0

        self.mistakes_list = np.zeros(n_samples)
        
        self.cum_mistakes_list = np.zeros(n_samples)

        self.z_t = np.zeros(self.hidden_units, dtype=np.float32)
        self.max_index = np.zeros(self.hidden_units, dtype=np.int32)

        for n_sample in xrange(n_samples):

            x_t = X[n_sample, :]
            y_t = y_true[n_sample]

            y_pred = self._fit_sample(x_t, y_t)

            if y_pred != y_t:
                counter += 1
            
            self.cum_mistakes_list[n_sample] = counter

            self.mistakes_list[n_sample] = counter * (100.0 / (n_sample + 1))

        self.cumulative_error = counter
        self.cumulative_error_rate = self.mistakes_list[-1]

    def _fit_sample(self, x_t, y_t, w_0=None, w_1=None):

        if w_0 is not None and w_1 is not None:
            self.w_0 = w_0
            self.w_1 = w_1

        x_t_norm = np.sqrt(1 + np.sum(x_t * x_t))

        x_t_normalized = x_t / x_t_norm

        # Perform forward through first layer
        max_out = np.dot(x_t_normalized, self.w_0[1:, :]) + (self.w_0[0, :] * (1.0 / x_t_norm))

        z_t = self.z_t
        max_index = self.max_index

        # Get max.value and max.index (MaxOut layer)
        for unit in xrange(self.hidden_units):
            base = unit * self.pieces

            # Argmax of the output for this unit
            max_index[unit] = base + np.argmax(max_out[base: base + self.pieces], axis=0)

            # Max value of the output for this unit
            z_t[unit] = max_out[max_index[unit]]

        z_t_mo = z_t.copy()

        #Normalize the projection
        z_t /= np.linalg.norm(z_t)

        #Predict
        f_t = np.dot(z_t.T, self.w_1)
        if f_t >= 0:
            y_pred = 1
        else:
            y_pred = -1

        #Suffer loss
        loss = 1.0 - y_t * f_t

        self.current_loss = loss

        if self.always_update or loss > 0:

            self.pa_classification(z_t, self.w_1, y_t, loss)

            self.pa_representation(self.w_1, z_t, y_t)

            for i in xrange(self.hidden_units):
                self.pa_regression(x_t_normalized, x_t_norm, z_t_mo[i], self.w_0[:, max_index[i]], z_t[i])

        return y_pred

    def fit_one_sample(self, x_t, y_t, w_0=None, w_1=None):

        if y_t != -1 and y_t != 1:
            raise Exception("Labels must be {-1,1}")

        if w_0 is not None and w_1 is not None:
            self.w_0 = w_0
            self.w_1 = w_1

        x_t_norm = np.sqrt(1 + np.sum(x_t * x_t))

        x_t_normalized = x_t / x_t_norm

        # Perform forward through first layer
        max_out = np.dot(x_t_normalized, self.w_0[1:, :]) + (self.w_0[0, :] * (1.0 / x_t_norm))

        z_t = np.zeros(self.hidden_units, dtype=np.float32)
        max_index = np.zeros(self.hidden_units, dtype=np.int32)

        # Get max.value and max.index (MaxOut layer)
        for unit in xrange(self.hidden_units):
            base = unit * self.pieces

            # Argmax of the output for this unit
            max_index[unit] = base + np.argmax(max_out[base: base + self.pieces], axis=0)

            # Max value of the output for this unit
            z_t[unit] = max_out[max_index[unit]]

        z_t_mo = z_t.copy()

        #Normalize the projection
        z_t /= np.linalg.norm(z_t)

        #Predict
        f_t = np.dot(z_t.T, self.w_1)

        if f_t >= 0:
            y_pred = 1
        else:
            y_pred = -1

        #Suffer loss
        loss = 1.0 - y_t * f_t

        self.current_loss = loss

        if self.always_update or loss > 0:

            self.pa_classification(z_t, self.w_1, y_t, loss)

            self.pa_representation(self.w_1, z_t, y_t)

            for i in xrange(self.hidden_units):
                self.pa_regression(x_t_normalized, x_t_norm, z_t_mo[i], self.w_0[:, max_index[i]], z_t[i])

        return y_pred

    def predict_batch(self, X, w_0=None, w_1=None):

        if w_0 is None and w_1 is None:
            w_0 = self.w_0
            w_1 = self.w_1

        n_samples = X.shape[0]

        y_pred = np.zeros(n_samples, dtype=np.int32)

        for n_sample in xrange(n_samples):
            x_t = X[n_sample, :]

            y_t_pred, _ = self.predict_one_sample(x_t, w_0, w_1)

            y_pred[n_sample] = self.inverse_label_mapping[y_t_pred]

        return y_pred

    def predict_one_sample(self, x_t, w_0=None, w_1=None):

        if w_0 is None and w_1 is None:
            w_0 = self.w_0
            w_1 = self.w_1

        x_t_norm = np.sqrt(1 + np.sum(x_t * x_t))

        x_t_normalized = x_t / x_t_norm

        # Perform forward through first layer
        max_out = np.dot(x_t_normalized, w_0[1:, :]) + (w_0[0, :] * (1.0 / x_t_norm))

        z_t = np.zeros(self.hidden_units, dtype=np.float32)

        # Get max.value and max.index (MaxOut layer)
        for unit in xrange(self.hidden_units):
            base = unit * self.pieces

            # Max value of the output for this unit
            z_t[unit] = np.max(max_out[base: base + self.pieces], axis=0)

        #Normalize the projection
        z_t /= np.linalg.norm(z_t)

        #Preditc
        f_t = np.dot(z_t.T, w_1)
        if f_t >= 0:
            y_pred = 1
        else:
            y_pred = -1

        return y_pred, f_t

    def project_one_sample(self, x_t, w_0=None, w_1=None):

        if w_0 is None and w_1 is None:
            w_0 = self.w_0
            w_1 = self.w_1

        x_t_norm = np.sqrt(1 + np.sum(x_t * x_t))

        x_t_normalized = x_t / x_t_norm

        # Perform forward through first layer
        max_out = np.dot(x_t_normalized, w_0[1:, :]) + (w_0[0, :] * (1.0 / x_t_norm))

        z_t = np.zeros(self.hidden_units, dtype=np.float32)

        # Get max.value and max.index (MaxOut layer)
        for unit in xrange(self.hidden_units):
            base = unit * self.pieces

            # Max value of the output for this unit
            z_t[unit] = np.max(max_out[base: base + self.pieces], axis=0)

        #Normalize the projection
        z_t_norm = z_t/np.linalg.norm(z_t)


        return z_t_norm, z_t

    def pa_classification(self, z_t, w_1_t, y_t, loss):

        if loss > 0:

            value = ((1.0 - self.alpha) * loss)

            if value < self.C_1:
                esc = value
            else:
                esc = self.C_1

            w_1_t += esc * z_t * y_t

    def pa_representation(self, w_1_t, z_t, y_t):

        y_pred = np.dot(w_1_t.T, z_t)

        loss = 1.0 - y_t * y_pred

        if loss > 0:

            w_1_t_mod_norm = self.get_mod_norm(w_1_t)

            esc = loss / w_1_t_mod_norm

            z_t += esc * w_1_t * y_t

    def pa_regression(self, x_t_normalized, x_t_norm, z_t_mo_i, w_0_t, z_t_i):

        loss = np.abs(z_t_i - z_t_mo_i) - self.epsilon

        if loss > 0:

            sig = np.sign(z_t_i - z_t_mo_i)

            if loss < self.C_0:
                esc = loss
            else:
                esc = self.C_0

            w_0_t[1:] += sig * esc * x_t_normalized
            w_0_t[0] += sig * esc * (1.0 / x_t_norm)

    def __str__(self):

        s = "Parameters: "
        s += "C_0= " + str(self.C_0)
        s += ", C_1= " + str(self.C_1)
        s += ", ALPHA= " + str(self.alpha)
        s += ", HU= " + str(self.hidden_units)
        s += ", PIECES= " + str(self.pieces)

        return s
