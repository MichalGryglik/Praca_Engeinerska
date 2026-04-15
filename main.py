from experiments.example_experiment import run as run_example_experiment
from experiments.tep_experiment import run as run_tep_experiment


def main():
    print("Framework loaded and modules detected.")
    run_tep_experiment()


if __name__ == "__main__":
    main()
