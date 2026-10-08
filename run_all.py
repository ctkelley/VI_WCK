"""Run the numerical experiments shown in Figures 2--4 of the paper."""
from experiments.run_example3 import main as run_example3
from run_section52_gamma1 import main as run_section52


def main():
    run_example3()
    run_section52()


if __name__ == "__main__":
    main()
