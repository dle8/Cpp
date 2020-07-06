/*
    Mutex: a mutual exclusion object that synchronizes access to a resource. It's a locking mechanism that make sures 1 thread can acquire mutex and enter critical section. This thread release mutex when it exits the critical sections.

    wait (mutex);
    …..
    Critical Section
    …..
    signal (mutex);

    Semaphore: signaling mechanism. an interger variable. and a thread that is waiting on a semaphore on a semaphore can be signaled by another thread. Two types: counting semaphore and binary semaphore

    wait(S) { while (S<=0); S--; }
    signal(S) { S++; }

    Mutex is different than a semaphore as it's a locking mechanism while a semaphore is a signalling mechanism. A binary semaphore can be used as a mutex. 
    Mutex also can only be signaled only by the thread that called the wait function.
*/