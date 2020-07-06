/*
    - When locks and condition variables are used together like this, the result is called a monitor :
        - A collection of procedures manipulating a shared data structure.
        - One lock that must be held whenever accessing the shared data (typically each procedure acquires the lock at the very beginning and releases the lock before returning).
        - One or more condition variables used for waiting.
    - There are other synchronization mechanisms besides locks and condition variables. Be sure to read about semaphores in the book or in the Pintos documentation.
*/