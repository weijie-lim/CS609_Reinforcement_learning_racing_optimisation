# CS609_Reinforcement_learning_racing_optimisation
CS609 Reinforcement Learning Project: Race Strategy Optimisation

Given the following race conditions, the project objective is to optimise timing and number of  pit stops:
- Number of laps: 162
- Race Track: circular
- Race Track Radius: 600m to 1200m
- Weather: Each weather type will result in varying maximum speeds and rates of degradation of the tyres
  1. Dry (eg. sunny day)
  2. 20% Wet (eg. drizzle which just started)
  3. 40% Wet
  4. 60% Wet
  5. 80% Wet
  6. 100% Wet (eg. heavy rain)
- Tyre Choices: Each tyre will have a different rate of degradation
  1. Ultrasoft
  2. Soft
  3. Intermediate
  4. Fullwet

Additional Details:

<img width="524" alt="image" src="https://github.com/weijie-lim/CS609_Reinforcement_learning_racing_optimisation/assets/47061871/16d14da3-5802-48b7-adef-f82d298c71d5">

Project Implementation:
- DQN
- Monte Carlo Tree Search
- Rules based

Project Evaluation Data Set:
- Our test set is a range of various weather sequences, approximately 90 different weather sequences, all beginning with an intermediate tyre choice.

Project Results

<img width="526" alt="image" src="https://github.com/weijie-lim/CS609_Reinforcement_learning_racing_optimisation/assets/47061871/7d19cb1f-0b86-498c-9696-1a44afc65eae">

- Notable behaviour
  - (-11,093) Rules-based will act immediately after valid weather changes
  - (-11,102) DQN shows heavy oscillation but is similar to rules-based behaviour. 
  - (-11,087) MCTS behaviour erratic but seems to react appropriately to weather changes

<img width="526" alt="image" src="https://github.com/weijie-lim/CS609_Reinforcement_learning_racing_optimisation/assets/47061871/7141139e-1503-4fcb-b464-0ca631800cc5">

- Notable behaviour
  - (-16,007) Rules-based will act immediately after valid weather changes.
  - (-16,009) DQN shows heavy oscillation but is like rules-based behaviour. 
  - (-16,014) MCTS behaviour erratic but seems to react appropriately to weather changes


<img width="526" alt="image" src="https://github.com/weijie-lim/CS609_Reinforcement_learning_racing_optimisation/assets/47061871/b1daa30e-386a-4241-b68b-52f8bd39ef2c">

- Notable behaviour
  - (-18,694) Rules-based responds to 60% Wet weather condition.
  - (-18,770) DQN shows heavy oscillation in tyres, which is hard to interpret. 
  - (-18,588) MCTS takes a different tyre trajectory compared to rules-based, using all 4 tyre types. The difference in results is also much larger here.


References
Piccinotti, D., Likmeta, A., Brunello, N., & Restelli, M. (2021). Online planning for F1 race strategy identiﬁcation - github pages. Online Planning for F1 Race Strategy Identification. https://prl-theworkshop.github.io/prl2021/papers/PRL2021_paper_1.pdf Taken from: Journal: Association for the Advancement of Artificial Intelligence (AAAI), www.aaai.org 

Project Mates Credit:
- Chia Dehan
- Colin Jiang
- Ng Juan Yong
- Lim Wei Jie
