/*
    SpMV/COO helps further reduce the amount of padding in ELL by sorting and partitioning the rows of the sparse matrix.

    Sort the row from longest to shortest, it looks like an inverted triangular matrix (Jagged Diagonal Storage (JDS) format).
    Once a sparse matrix is in the JDS format, parition matrix into section of rows. After sorted, all rows in the section will likely
    have similar numbers of non zero elements. An ELL representation for each section can be generated. Within each section, only need
    to pad the rows to match the row with the max number of elements in that section -> reduce further padding overhead.

        For example:
            [                               [                               [
                3, 0, 1, 0                      3, 1, *                         2, 4, 1
                0, 0, 0, 0                      *, *, *                         3, 1, *
                0, 2, 4, 1                      2, 4, 1                         1, 1, *
                1, 0, 0, 1                      1, 1, *                         *, *, *
            ]                               ]                               ]
                                            <CSR with padding>                  <JDS>

            data[7 ]           = {2, 4, 1, 3, 1, 1, 1}
            col_index[7]       = {1, 2, 3, 0, 2, 0, 3}
            Jds_row_index[4]   = {2, 0, 3, 1}
            Jds_section_ptr[4] = {0, 3, 7, 7}

            Jds_section_ptr[i]: begining of row i in data
*/