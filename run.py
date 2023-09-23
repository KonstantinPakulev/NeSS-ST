from omegaconf import OmegaConf
import multiprocessing
import hydra

from source.experiment import SummertimeExperiment

# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

@hydra.main(config_path='config', version_base='1.1')
def run(config):
    print("\n")
    print("Configuration file:")
    print("-" * 60)
    print(OmegaConf.to_yaml(config))
    print("-" * 60)

    OmegaConf.set_struct(config, False)

    experiment = SummertimeExperiment(config)
    experiment.run()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    run()
