import os
import sys
import zipfile
import subprocess

sys.path.append('./Tool')
import Tool.Info
import Tool.Checkout
import Tool.RunTest


def checkout(destination_directory, time, difficulty, contest, task, language, cwd):
    if destination_directory is None:
        print("Error: The -w parameter is required.")
        print("Use -h for help.")
        return

    print(f"Checkout command called with:")
    print(f"destination_directory = {destination_directory}")
    print(f"time = {time}")
    print(f"difficulty = {difficulty}")
    print(f"contest = {contest}")
    print(f"task = {task}")
    print(f"language = {language}")

    try:
        Tool.Checkout.checkout(destination_directory, time, difficulty, contest, task, language, cwd)
    except KeyError as e:
        print(f"Error: Task '{e.args[0]}' not found. Please check your date.txt or difficulty.txt.")
    except Exception as e:
        print(f"Unexpected error during checkout: {e}")


def run_tests(dest_dir, task_name, repo_dir, coverage=False):
    command = [
        "python", os.path.join(repo_dir, "ConDefects.py"), "run",
        "-w", dest_dir,
        "-s", task_name
    ]
    if coverage:
        print("Warning: --coverage argument is not supported. Skipping.")
    print(f"Running tests: {command}")
    try:
        subprocess.run(command, check=True, cwd=repo_dir)
        print(f"Tests for task {task_name} completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error during test execution: {e}")
        print(f"Command: {e.cmd}")
        print(f"Return code: {e.returncode}")
        exit(1)


def main(cwd):
    import argparse

    parser = argparse.ArgumentParser(description='ConDefects command-line tool.')
    subparsers = parser.add_subparsers(dest='command')

    # checkout command
    checkout_parser = subparsers.add_parser('checkout')
    checkout_parser.add_argument('-w', '--dest-dir', required=True)
    checkout_parser.add_argument('-t', '--time', nargs=2)
    checkout_parser.add_argument('-d', '--difficulty', nargs=2, type=int)
    checkout_parser.add_argument('-c', '--contest')
    checkout_parser.add_argument('-s', '--task')
    checkout_parser.add_argument('-l', '--language', choices=['java', 'python'])

    # run command
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('-w', '--dest-dir', required=True)
    run_parser.add_argument('-s', '--task')
    run_parser.add_argument('-t', '--test', nargs='+')

    args = parser.parse_args()

    if args.command == 'checkout':
        checkout(args.dest_dir, args.time, args.difficulty, args.contest, args.task, args.language, cwd)
    elif args.command == 'run':
        run_tests(args.dest_dir, args.task, cwd, coverage=False)
    else:
        print("Invalid command. Use -h for help.")


if __name__ == '__main__':
    cwd = os.getcwd()
    testPath = os.path.join(cwd, 'Test')
    testZipPath = os.path.join(cwd, 'Test.zip')

    if not os.path.exists(testPath) and os.path.exists(testZipPath):
        print('Extracting Test.zip...')
        with zipfile.ZipFile(testZipPath, 'r') as zip_ref:
            zip_ref.extractall(cwd)
        print('Extracting Test.zip finished.')
    elif not os.path.exists(testPath) and not os.path.exists(testZipPath):
        print('Test.zip not found.')
        print("Don't worry! You can download it by following the guide on our GitHub repository.")
        print("Then make sure to place the downloaded Test.zip in the same directory as ConDefects.py.")
        exit()

    main(cwd)
