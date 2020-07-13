/*
    Assume graph is represented in the CSR format. The function receives index of the source vertex, the edges array (edges),
    the destination array (dest), label array (label) whose elements store the visit status information for the vertices.

    Two frontier arrays:
        - One stores the frontier vertices discovered in the previous iteration (previous frontier), and one stores the frontier 
        vertices that are being discovered in the current iteration (current frontier). These arrays are frontier[0][MAX_FRONTIER_SIZE]
        and frontier[1][MAX_FRONTIER_SIZE].
        - PING-PONG BUFFERING: 
            - Roles of these two arrays alternate. By switching the roles of these two arrays, we avoid the need for copying the contents
            from a current frontier array to a previous frontier array when we move to the next iteration.
            - c_frontier: points to the beginning of the current frontier array. p_frontier: points to the beginning of the previous
            frontier array.
            - p_frontier_tail: number of elements inserted into the previous frontier array. c_frontier_tail: number of frontier vertices
            that have been inserted into the current frontier array so far
*/

void BFS_sequential(int source, int *edges, int *dest, int *label) {
    int frontier[2][MAX_FRONTIER_SIZE];
    int *c_frontier = &frontier[0], *p_frontier = &frontier[1];
    int c_pointer_tail = 0, p_frontier_tail = 0;

    insert_frontier(source, p_frontier, &p_frontier_tail);
    label[source] = 0;

    while (p_frontier_tail > 0) {
        for (int f = 0; f < p_frontier_tail; ++f) { // Visit all previous frontier vertices
            int c_vertex = p_frontier[f]; // Pick up 1 of the previous frontier vertex
            for (int i = edges[c_vertex]; i < edges[c_vertex + 1]; ++i) { // for all its edge
                if (label[dest[i]] == -1) { // The vertex has not been visited
                    insert_frontier(dest[i], c_frontier, &c_frontier_tail);
                    label[dest[i]] = label[c_vertex] + 1;
                }
            }
        }
        swap(c_frontier, p_frontier); // swap previous and current - ping-pong buffer
        p_frontier_tail = c_frontier_tail;
        c_frontier_tail = 0;
    }
}