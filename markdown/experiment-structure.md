### Experiment structure

- Keeping track of your experiments and model iterations is **super important**! Most of the code written for an ML pipeline is actually about managing your experiments in a way where you stay sane. 
- Keep each experiment (trained model) in a separate place, and save all of the information you need to reproduce the experiment alongside it (notes, hyperparameter values, etc.) I usually have an `experiments/` folder that contains files like this:
 	- `experiment_1/`
		- `hyperparameters.json`
		- `checkpoints/` (saved model weight checkpoints go in here)
		- `notes.txt` (general notes and comments about this experiment)
- Here is a [blog post](https://andrewhalterman.com/2018/03/05/managing-machine-learning-experiments/) of a similar approach by Andy Halterman.
