import retro
import gym

env = retro.make('SuperMarioBros-Nes')
obs = env.reset()

print(obs.shape)

done = False
while not done:
    obs, reward, done, info = env.step(env.action_space.sample())
    env.render()


env.close()
