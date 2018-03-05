import threading
import os
import queue
import time
from trainerholder import TrainHolder


def run(queueLock, workQueue, trainh, exitFlag, shuffle, seed, debug_mode):
    while not exitFlag[0]:
        queueLock.acquire()
        if not workQueue.empty():
            folder_name = workQueue.get()
            queueLock.release()
            trainh.train(folder_name, shuffle, seed, debug_mode)
            workQueue.task_done()
        else:
            queueLock.release()
            time.sleep(1)


def parallel_train(trainh: TrainHolder, folder_name=None, folder_list=None, max_threads=None, shuffle=True, seed=None, debug_mode=False):
    """
    folder_name : str
    folder_list : str
        find all classes in folder_name and/or one by one in folder_list

    max_threads : int
        max number of threads for training
        each thread trains one class

    debug_mode : bool
        prints debug info
    """

    fn, fl = folder_name is None, folder_list is None
    if fn and fl:
        print('No folder specified')
        return

    path_list = []

    if not fn:
        for dirname in os.listdir(folder_name):
            subfolder = os.path.join(folder_name, dirname)
            if not os.path.isdir(subfolder):
                continue
            path_list.append(subfolder)
    if not fl:
        for subfolder in folder_list:
            if not os.path.isdir(subfolder):
                continue
            path_list.append(subfolder)

    num_threads = len(path_list) if max_threads is None else min(
        max_threads, len(path_list))
    queueLock = threading.Lock()
    workQueue = queue.Queue()
    exitFlag = [False]

    print('Threads Creating')
    threadlist = [threading.Thread(target=run, args=(queueLock, workQueue, trainh, exitFlag, shuffle, seed, debug_mode))
                  for i in range(num_threads)]
    for var in threadlist:
        #var.daemon = True
        var.start()

    print('Threads started')
    print(len(path_list))
    queueLock.acquire()
    for folder_name in path_list:
        workQueue.put(folder_name)
    queueLock.release()

    workQueue.join()

    exitFlag[0] = True
    for t in threadlist:
        t.join()
    print('Training is fully completed!')
