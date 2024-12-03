import os
import pickle
import shutil

def checkout(destination_directory, time, difficulty, contest, taskR, language, cwd):
    srcPath = os.path.join(cwd, "Code")
    testPath = os.path.join(cwd, "Test")
    taskList = os.listdir(srcPath)

    # Ensure the destination directory exists
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Load the date dictionary
    datePath = os.path.join(cwd, "Tool", "date")
    try:
        with open(datePath, "rb") as f:
            date = pickle.load(f)
    except Exception as e:
        print(f"Error loading date file: {e}")
        return

    # Filter tasks by time
    if time is not None:
        time1, time2 = time
        if time1 > time2:
            print("Invalid time range.")
            return
        for task in reversed(taskList):
            thisTime = date.get(task)
            if thisTime is None or thisTime < time1 or thisTime > time2:
                taskList.remove(task)

    # Load the difficulty dictionary
    diffPath = os.path.join(cwd, "Tool", "difficulty")
    try:
        with open(diffPath, "rb") as f:
            diff = pickle.load(f)
    except Exception as e:
        print(f"Error loading difficulty file: {e}")
        return

    # Filter tasks by difficulty
    if difficulty is not None:
        diff1, diff2 = map(int, difficulty)
        if diff1 > diff2:
            print("Invalid difficulty range.")
            return
        for task in reversed(taskList):
            thisDiff = diff.get(task)
            if thisDiff is None or int(thisDiff) < diff1 or int(thisDiff) > diff2:
                taskList.remove(task)

    # Filter tasks by contest
    if contest is not None:
        for task in reversed(taskList):
            if contest not in task:
                taskList.remove(task)

    # Filter by specific task
    if taskR is not None:
        taskList = [task for task in taskList if task == taskR]

    # Copy filtered tasks
    for task in taskList:
        taskPath = os.path.join(srcPath, task)
        taskDestPath = os.path.join(destination_directory, task)
        if not os.path.exists(taskDestPath):
            os.makedirs(taskDestPath)
        for inner in os.listdir(taskPath):
            innerPath = os.path.join(taskPath, inner)
            innerDestPath = os.path.join(taskDestPath, inner)
            if language is None or innerPath.lower().endswith(language.lower()):
                shutil.copytree(innerPath, innerDestPath, dirs_exist_ok=True)
