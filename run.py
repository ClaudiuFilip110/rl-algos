import retro
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

model = PPO.load('tmp/best_model.zip')


def main():
    env = retro.make('SuperMarioBros-Nes')
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        env.render()


if __name__ == '__main__':
    main()
