from argparse import ArgumentParser

from antifraud.app import main


def run():
    parser = ArgumentParser(description='Antifraud pipeline runner')
    parser.add_argument('--task', type=int, choices=[1, 2, 3, 4], help='Run only one task (1..4)')
    args = parser.parse_args()
    main(task=args.task)
