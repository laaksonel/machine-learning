# CartPole OpenAI Gym - Reinforcement learning

https://gym.openai.com/envs/CartPole-v1/

## Episode
Episode ends when the pole is more than 15 degrees from vertical axis or the cart is more than 2.4 units from the center.

## States

The states of the cart pole are:
- cart position
- cart velocity
- pole angle (in radians)
- pole tip velocity

## Actions
The cart can be moved to left or right.
- left = 0
- right = 1

## Training

The agent is trained with traditional Q-learning
[<img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FQ-learning&psig=AOvVaw3-CzC9j6Fhckec35G5LQdU&ust=1602401806282000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCMDKroXCqewCFQAAAAAdAAAAABAE">](Q-learning formula)

### States and continuous values
As you can see, the states are all continuous values. In order to our model to ever converge, we need to discretize (assign to interval segments) the states. We cannot possibly model all of our model/action combinations with continuous values.

All the values in  the same interval segment are treated as same, this way we have bins where each continuous value falls. The idea of this discretization is to make the Q-table smaller and manageable since we have reasonable count of states.

Apart from the discretized values, the model is trained with traditional Q-table.
Learning rate and "exploration" multiplier are adjusted during the training. Both of the terms will decrease towards the
end of the training. The previous experiences are thus weighted higher in the later episodes. The same way, the
exploration of the environment is decreased and previous knowledge is exploited more to get the desired behaviour (cart pole staying in balance).

