from experiments.sine_process_experiment import run as run_sine_process_experiment
from experiments.example_experiment import run as run_example_experiment
from experiments.tep_experiment import run as run_tep_experiment


def main():
    run_sine_process_experiment()
    run_example_experiment()
    run_tep_experiment()


if __name__ == "__main__":
    main()
