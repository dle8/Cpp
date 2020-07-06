/*
    Synchronize mechanisms based on particular condition to become true

    - wait(condition, lock): release lock, put thread to sleep until condition is signaled; when thread wakes up again, re-acquire lock before returning.
    - signal(condition, lock): if any threads are waiting on condition, wake up one of them. Caller must hold lock, which must be the same as the lock used in the wait call.
    - broadcast(condition, lock): same as signal, except wake up all waiting threads.
    Warning: when a thread wakes up after cond_wait there is no guarantee that the desired condition still exists: another thread might have snuck in.
*/
const int SIZE = 64;

char buffer[SIZE];
int count = 0, putIndex = 0, getIndex = 0;
struct lock l;
struct condition dataAvailable;
struct condition spaceAvailable;

void lock_init(lock *l);
void cond_init(condition *dataAvailable);
void cond_init(condition *spaceAvailable);

void cond_wait(condition *dataAvailable, lock *l);
void cond_signal(condition *spaceAvailable, lock *l);

void lock_init(lock *l) {}
void lock_acquire(lock *l) {}
void lock_release(lock *l) {}

void put(char c) {
    lock_acquire(&l);
    while (count == SIZE) {
        cond_wait(&spaceAvailable, &l);
    }
    count++;
    buffer[putIndex] = c;
    putIndex++;
    if (putIndex == SIZE) {
        putIndex = 0;
    }
    cond_signal(&dataAvailable, &l);
    lock_release(&l);
}

char get() {
    char c;
    lock_acquire(&l);
    while (count == 0) {
        cond_wait(&dataAvailable, &l);
    }
    count--;
    c = buffer[getIndex];
    getIndex++;
    if (getIndex == SIZE) {
        getIndex = 0;
    }
    cond_signal(&spaceAvailable, &l);
    lock_release(&l);
    return c;
}