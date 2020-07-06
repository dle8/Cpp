/*
    lock: an object can be owned by a single thread at any given time. Basic operations on lock:
        - acquire: mark the lock as owned by the current thread. wait if other threads own the lock. There's a queue to keep track of multiple waiting threads
        - release: mark the lock as free.
*/

/*
    A more complex example: producer/consumer.
        - Producers add characters to a buffer
        - Consumers remove characters from the buffer
        - Characters will be removed in the same order added
*/

const int SIZE = 64;

char buffer[SIZE];
int count = 0, putIndex = 0, getIndex = 0;
struct lock l;

void lock_init(lock *l) {};

void lock_acquire(lock *l) {}

void lock_release(lock *l) {}

void put(char c) {
    lock_acquire(&l);
    while (count == SIZE) {
        lock_release(&l);
        lock_acquire(&l);
    }
    count++;
    buffer[putIndex] = c;
    putIndex++;
    if (putIndex == SIZE) {
        putIndex = 0;
    }
    lock_release(&l);
}

char get() {
    char c;
    lock_acquire(&l);
    while (count == 0) {
        lock_release(&l);
        lock_acquire(&l);
    }
    count--;
    c = buffer[getIndex];
    getIndex++;
    if (getIndex == SIZE) {
        getIndex = 0;
    }
    lock_release(&l);
    return c;
}

int main() {

}