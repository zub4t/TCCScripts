import numpy as np
import matplotlib.pyplot as plt
import lms_and_gradient_descend
import lms_and_gradient_descend_mini_batch
import bayesian_grid


lms_and_gradient_descend_mini_batch_errors = np.array(lms_and_gradient_descend_mini_batch.get_error())
lms_gradient_descend_errors = np.array(lms_and_gradient_descend.get_error())
bayesian_grid_errors = np.array(bayesian_grid.get_error())

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

arrays = [lms_and_gradient_descend_mini_batch_errors, lms_gradient_descend_errors, bayesian_grid_errors]
arrays_name=['lms_and_gradient_descend_mini_batch','lms_gradient_descend','bayesian_grid']
for array in arrays:
    count, bins_count = np.histogram(array, bins=50)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], cdf, label=arrays_name.pop(0))
plt.xlabel('Distance Error')
plt.ylabel('Cumulative Frequency')
plt.legend()

plt.savefig("CDF.png")
plt.show()