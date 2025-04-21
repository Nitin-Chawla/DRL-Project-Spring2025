# Install Dependencies
    conda create -f environment.yml

# Change Environment Reward Structure
Replace `rocket_base_env.py` and `rocket_landing_env.py` in `(your_env_lib)\site-packages\PyFlyt\gym_envs\rocket_envs\`

# To Perform Vertical Only Landing
* Change `randomize_drop` to False in rocket_landing_env.py (line: options = dict(randomize_drop=True, accelerate_drop=True))

# Running the Policy
* Run the test.py file to view policy in action. By default it runs the RND policy