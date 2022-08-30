import numpy as np


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        # Note: This was float64 earlier, but we changed it to float32
        self.mean = np.zeros(shape, "float32")
        self.var = np.ones(shape, "float32")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0, dtype="float32")
        batch_var = np.var(x, axis=0, dtype="float32")
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def __str__(self):
        return "Mean: %s, Var: %s, Count: %i" % (
            str(list(self.mean)),
            str(list(self.var)),
            self.count,
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta, dtype="float32") * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
