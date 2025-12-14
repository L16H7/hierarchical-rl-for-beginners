# Lower T1 Goalie Penalty Kick Environment
Original documentation: https://competesai.com/environments/env_T9LZoQ9GsQBZ

## Description

Penalty Kick with Goalie (Lower T1) is a task that requires executing a penalty kick against a moving goalie in a soccer environment. The robot must kick the ball beyond the goalie obstacle, into the opponent's goal.

The task demands precise positioning to close in on the ball and dynamic control to deliver an effective and well-aimed kick, all within a continuous and high-dimensional action and observation space.

### Action Space

```
Box(shape=(12,), low=[-45,-45,-30,-65,-24,-15,-45,-45,-30,-65,-24,-15], high=[45,45,30,65,24,15,45,45,30,65,24,15])
```

### Observation Space

```
Box(shape=(45,), low=-inf, high=inf, float32)
```

### Import

```python
gym.make("LowerT1GoaliePenaltyKick-v0")
```

### Actions

The action space is a continuous vector of shape (12,), where each dimension corresponds to a joint torque command for the T1's articulated parts. The table below describes each dimension, interpreted by the joint torque controllers to compute joint commands.

| Index | Action              |
|-------|---------------------|
| 0     | hip_y_left torque   |
| 1     | hip_x_left torque   |
| 2     | hip_z_left torque   |
| 3     | knee_y_left torque  |
| 4     | ankle_y_left torque |
| 5     | ankle_x_left torque |
| 6     | hip_y_right torque  |
| 7     | hip_x_right torque  |
| 8     | hip_z_right torque  |
| 9     | knee_y_right torque |
| 10    | ankle_y_right torque|
| 11    | ankle_x_right torque|

To understand how the indices can be grouped to understand actions at a higher level, refer to the descriptions below:

#### Left Leg Control

**Description:** Controls the left leg joints including hip, knee, and ankle for walking, balance and scoring a goal.

**Indices:** 0-5

#### Right Leg Control

**Description:** Controls the right leg joints including hip, knee, and ankle for walking, balance and scoring a goal.

**Indices:** 6-11

### Observations

The observation is a ndarray with shape (45,) where the values corresponding to the following features:

#### Joint Positions

**Description:** The absolute positions of the robot's body parts (the right and left leg joints).

**Indices:** 0-11

#### Joint Velocities

**Description:** The absolute velocities of the robot's body parts (the right and left leg joints).

**Indices:** 12-23

#### Ball Position (Relative to Robot)

**Description:** The relative position of the ball relative to the robot $(x, y, z)$ coordinates.

**Indices:** 24-26

#### Ball Linear Velocity (Relative to Robot)

**Description:** The linear velocity of the ball relative to the robot $(x, y, z)$ coordinates.

**Indices:** 27-29

#### Ball Angular Velocity (Relative to Robot)

**Description:** The rotational velocity of the ball relative to the robot $(x, y, z)$ coordinates.

**Indices:** 30-32

#### Goal Position (Relative to Robot)

**Description:** A vector from the robot to the center of the goal $(x, y, z)$ coordinates.

**Indices:** 33-35

#### Goal Position (Relative to Ball)

**Description:** A vector from the ball to the center of the goal $(x, y, z)$ coordinates.

**Indices:** 36-38

#### Goalkeeper Position (Relative to Robot)

**Description:** The relative position of the goalkeeper relative to the robot $(x, y, z)$ coordinates.

**Indices:** 39-41

#### Goalkeeper Linear Velocity (Relative to Robot)

**Description:** The linear velocity of the goalkeeper relative to the robot $(x, y, z)$ coordinates.

**Indices:** 42-44
