# Usage

## Point Mass (No Obstacle)
Use default arguments.
```
from rl_utils.envs import create_vectorized_envs
envs = create_vectorized_envs(
    "PointMass-v0", num_envs=32
)
```

Change default arguments.
```
from rl_utils.envs import create_vectorized_envs
from rl_utils.envs.pointmass import PointMassParams
envs = create_vectorized_envs(
    "PointMass-v0", num_envs=32, params=PointMassParams(dt=0.1, ep_horizon=10)
)
```

## Point Mass (With Obstacle)
Use default arguments.
```
from rl_utils.envs import create_vectorized_envs
envs = create_vectorized_envs("PointMassObstacle-v0", num_envs=32)
```
Change default arguments.
```
from rl_utils.envs import create_vectorized_envs
from rl_utils.envs.pointmass import PointMassObstacleParams
envs = create_vectorized_envs(
    "PointMassObstacle-v0",
    num_envs=32,
    params=PointMassObstacleParams(
        dt=0.1, ep_horizon=10, square_obstacles=[([0.5, 0.5], 0.11, 0.5, 45.0)]
    ),
)
```
