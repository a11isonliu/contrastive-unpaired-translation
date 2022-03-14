from GPUtil import showUtilization
from numba import cuda

print("Initial GPU Usage")
showUtilization()