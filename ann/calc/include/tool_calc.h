#ifndef TOOL_CALC_INCLUDE_TOOL_CALC_H
#define TOOL_CALC_INCLUDE_TOOL_CALC_H

#include "tool_calc/define.h"
#include <cstring>
#include <cstdint>

namespace tool 
{
namespace ann
{
namespace calc
{

/*
dimension : dim % 4 == 0 and dim >= 4
xxx ideal : dim is 2 ^ x and dim >= 4, size = dim * sizeof(v)
 *
*/

/* malloc sizeof(float) * dimension, align by cache line */
void* alloc_block(int size);
void free_block(void* block);

/* malloc sizeof(float) * dimension, align by cache line */
inline float* alloc_vector(int dimension) {
    return static_cast<float*>(alloc_block(dimension * sizeof(float)));
}

inline int8_t* alloc_vector_int8(int dimension) {
    return static_cast<int8_t*>(alloc_block(dimension * sizeof(int8_t)));
}

inline char* alloc_vector_ideal(int size) {
    return static_cast<char*>(alloc_block(size));
}

inline void free_vector(float* block) {
    free_block(static_cast<void*>(block));
}

inline void free_vector_int8(int8_t* block) {
    free_block(static_cast<void*>(block));
}

inline void free_vector_ideal(char* block) {
    free_block(static_cast<void*>(block));
}

/* malloc sizeof(float) * dimension * rows, align by cache line */
inline float* alloc_matrix(int dimension, int rows) {
    return static_cast<float*>(alloc_block(rows * dimension * sizeof(float)));
}

inline float* alloc_matrix_int8(int dimension, int rows) {
    return static_cast<float*>(alloc_block(rows * dimension * sizeof(int8_t)));
}

inline char* alloc_matrix_ideal(int size, int rows) {
    return static_cast<char*>(alloc_block(rows * size));
}

inline void free_matrix(float* block) {
    free_block(static_cast<void*>(block));
}

inline void free_matrix_int8(int8_t* block) {
    free_block(static_cast<void*>(block));
}

inline void free_matrix_ideal(char* block) {
    free_block(static_cast<void*>(block));
}

/* number of threads(openmp) */
void set_num_threads(int number);
int get_max_threads();

void set_epsinon(float eps);

/* get sum(i^2) by vector */
float get_l2(int dimension, float* vector);

/* normalize the vector/matrix, let L2 = 1 for each row */
void normalize_vector(int dimension, float* vector);
void normalize_matrix(int dimension, int rows, float* matrix);

/* dot v1 to v2, output to dot */    
void dot_vectors(int dimension, float *v0, float *v1, float *dot);

/* dot matrix to vector, output to dots */    
void dot_matrix_vector(int dimension, int rows, float *matrix, float *vector, float *dots);

/* dot matrix to matrix */
void dot_matrix_matrix(int dimension, int a_rows, int b_rows, float* a_matrix, float* b_matrix, float* dots);
void dot_matrix_matrix_int8(int dimension, int a_rows, int b_rows, int8_t* a_matrix, int8_t* b_matrix, int32_t* dots); 

/* L2 distance, v1 to v2 */
void dis_l2_vectors(int dimension, float *v0, float *v1, float *dis);

/* L2 distance, matrix to vector */    
void dis_l2_matrix_vector(int dimension, int rows, float *matrix, float *vector, float *dises);

/* L2 distance, matrix to vector, with buffer, sizeof(buf) >= dim */    
void dis_l2_matrix_vector_buf(int dimension, int rows, float *matrix, float *vector, float *dises, float* buffer);

/* get vector pointer for matrix(rowid) */
inline float* peek_vector(int dimension, int rowid, float* matrix) {
    return matrix + dimension * rowid;
}
inline int8_t* peek_vector_int8(int dimension, int rowid, int8_t* matrix) {
    return matrix + dimension * rowid;
}
inline char* peek_vector_ideal(int size, int rowid, char* matrix) {
    return matrix + size * rowid;
}

/* copy vector data from vector to matrix(rowid) */
void put_vector(int dimension, int rowid, float* matrix, float* vector);
void put_vector_int8(int dimension, int rowid, int8_t* matrix, int8_t* vector);
inline void put_vector_ideal(int size, int rowid, char* matrix, char* vector) {
    memcpy(peek_vector_ideal(size, rowid, matrix), vector, size);
};

/* copy vector data from matrix(rowid) to vector */
void get_vector(int dimension, int rowid, float* matrix, float* vector);
void get_vector_int8(int dimension, int rowid, int8_t* matrix, int8_t* vector);
inline void get_vector_ideal(int size, int rowid, char* matrix, char* vector) {
    memcpy(vector, peek_vector_ideal(size, rowid, matrix), size);
}

/* vector_y = vector_y + vector_x * scalar */
void axpy_vector(int dimension, float scalar, float* vector_x, float* vector_y);
/* vector = vector * scalar */
void product_vector(int dimension, float scalar, float* vector);

/* normalize(center * scalar + matrix rows) -> center */
void product_center(int dimension, int rows, float *matrix, int scalar, float* center);
inline void get_center(int dimension, int rows, float *matrix, float* center) {
    product_center(dimension, rows, matrix, 0, center);
}

} // namespace calc
} // namespace ann
} // namespace tool

#endif /* TOOL_CALC_INCLUDE_TOOL_CALC_H */

