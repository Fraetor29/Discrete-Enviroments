# Discrete-Enviroments

Here is presented DQN algorithm
In DQN_Main we can change parameters to see how Agent learns.

Parameter 'egreedy_decay' is to adjust the exponential probability to pick random action. This is good if we want our Agent to explore more.
If we set a higher number of training episodes, we can choose a higher value for 'egreedy_decay'.

Down below is presented how it works.      


![bla2](https://user-images.githubusercontent.com/102504166/161492069-ae3b2cbf-d559-4b61-9ad1-25254f195557.png)

Here are presented the results of training.

![bla1](https://user-images.githubusercontent.com/102504166/161492302-f73cd840-591d-41fb-a9f2-3db1576af38c.png)

The 'train()' method can return the average rewards and we can store it in some variables. Thus we can train multiple Agents with different parameters and compare their evolutions.
