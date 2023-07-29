import safety_gymnasium
from sbx.wrappers.monitor import SafeMonitor

from sbx.droq.droqlag import DroQLag

env = safety_gymnasium.make("SafetyPointGoal1-v0", render_mode='rgb_array')
env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
env = SafeMonitor(env)

model = DroQLag("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000, progress_bar=True)

