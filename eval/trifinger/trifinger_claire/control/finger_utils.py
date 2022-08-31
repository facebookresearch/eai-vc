import numpy as np

# Here, hard code the base position of the fingers (as angle on the arena)
r = 0.15
theta_0 = 90
theta_1 = 310
theta_2 = 200
# theta_2 = 3.66519 # 210 degrees

FINGER_BASE_POSITIONS = [
    np.array(
        [[np.cos(theta_0 * (np.pi / 180)) * r, np.sin(theta_0 * (np.pi / 180)) * r, 0]]
    ),
    np.array(
        [[np.cos(theta_1 * (np.pi / 180)) * r, np.sin(theta_1 * (np.pi / 180)) * r, 0]]
    ),
    np.array(
        [[np.cos(theta_2 * (np.pi / 180)) * r, np.sin(theta_2 * (np.pi / 180)) * r, 0]]
    ),
]


# These are offsets to cube half size for computing contact points
# Tuned empirically; do not match actual fingertip dimensions
FT_RADIUS = 0.008  # PRO 
#FT_RADIUS = 0.007 # EDU
