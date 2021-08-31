# Graph Attention Networks for Traffic Signal Control

To setup the environment with the necessary packages to run the simulation, follow the instructions below.
1. Download and install [SUMO](https://sumo.dlr.de/docs/Downloads.php) (If you have Flow, you already have SUMO installed).
2. Create and activate a conda environment and install the dependencies within the environment. This
can be done by running the below command.

3. We use Weights and Biases to track the progress of our training. To do this, first create a free account
and log into the Weights and Biases website. You will see a dashboard. Then in terminal, run,
```
wandb login
```

This will prompt you a login code (it will open a new browser window and show you the login code) which you can then copy and paste into the terminal as it instructs.

When you start your training, you can simply visit Weights and Biases website and there will be a project named DQN ATL. Click on that project and you will see all the runs that you have run so far. Click on one of such runs to see the detailed plots corresponding to that run. 


To train a model (1 × 7 grid of intersections)
```
python training_main.py
```

A recoded video of a trained agent (1 traffic light) can be found [here](https://www.dropbox.com/s/7ls4g7v49cdjdg3/video.mov?dl=0). 
