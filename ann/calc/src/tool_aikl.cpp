#include <cmath>
#include <cstdlib>
#include <cstring>
#include "tool_calc.h"

#ifdef TOOL_CALC_MKL
#include "mkl.h"
#include "oneapi/dnnl/dnnl.h"
#endif

#ifdef TOOL_CALC_OPENMP
#include <omp.h>
#endif

#ifdef TOOL_CALC_ARM
#include <arm_neon.h>
#endif

namespace tool 
{
namespace ann
{
namespace calc
{

static float s_epsinon = TOOL_CALC_EPSINON;

#ifdef TOOL_CALC_MKL   
void* alloc_block(int size) {
    void* ptr = mkl_malloc(size, 64);
    if (ptr == NULL) {
        return nullptr;
    }
    return ptr;
}

void free_block(void* block) {
    mkl_free(block);
}

#else
void* alloc_block(int size) {
    void* ptr = malloc(size);
    if (ptr == NULL) {
        return nullptr;
    }
    return ptr;
}

void free_block(void* block) {
    free(block);
}

#endif

#if defined(TOOL_CALC_MKL)
void set_num_threads(int number) {
    mkl_set_num_threads(number);
}

int get_max_threads() {
    return mkl_get_max_threads();
}
 
#elif defined(TOOL_CALC_OPENMP)
void set_num_threads(int number) {
    omp_set_num_threads(number);
}

int get_max_threads() {
    return omp_get_max_threads();
}

#else
void set_num_threads(int number) {
    (void)number;
}

int get_max_threads() {
    return 1;
}

#endif

void set_epsinon(float epsinon) {
    s_epsinon = epsinon;
}

#ifdef TOOL_CALC_MKL
float get_l2(int dimension, float* vector) {
    int incx = 1;
    float nrm = cblas_snrm2(dimension, vector, incx);
    if (nrm <= s_epsinon) {
        return 0.0;
    }
    return nrm;
}

void normalize_vector(int dimension, float* vector) {
    int incx = 1;
    float nrm = cblas_snrm2(dimension, vector, incx);
    if (nrm <= s_epsinon) {
        vector[0] = 1.0;
        return;
    }
    cblas_sscal(dimension, 1.0 / nrm, vector, incx);
    return;
}

void normalize_matrix(int dimension, int rows, float* matrix) {
    for (int idx = 0; idx < rows; ++idx) {
        normalize_vector(dimension, peek_vector(dimension, idx, matrix));
    }
    return;
}

#else
#ifdef TOOL_CALC_ARM
// TODO: dimension % 4 != 0
float get_l2(int dimension, float* vector) {
    float nrm = 0.0;
    float32x4_t x4_sum = vdupq_n_f32(0.0f);
    for (int idx = 0; idx < dimension; idx += 4) {
        x4_sum = vmlaq_f32(x4_sum, vld1q_f32(vector + idx), vld1q_f32(vector + idx));
    }
    float32x2_t _ss = vadd_f32(vget_high_f32(x4_sum), vget_low_f32(x4_sum));
    nrm += vget_lane_f32(vpadd_f32(_ss, _ss), 0);
    nrm = sqrtf(nrm);
    return nrm;
}

// TODO: dimension % 4 != 0
void normalize_vector(int dimension, float* vector) {
    float nrm = get_l2(dimension, vector);
    if (nrm <= s_epsinon) {
        vector[0] = 1.0;
        return;
    }
    float dnrm = 1.0 / nrm;

    float32x4_t _nrm = vdupq_n_f32(dnrm);
    for (int idx = 0; idx < dimension; idx += 4) {
        vst1q_f32(vector + idx, vmulq_f32(vld1q_f32(vector + idx), _nrm));
    }
    return;
}

#else
float get_l2(int dimension, float* vector) {
    float nrm = 0.0;
    for (int idx = 0; idx < dimension; ++idx) {
        float v = vector[idx];
        nrm += v * v;
    }
    if (nrm <= s_epsinon) {
        return 0.0;
    }
    nrm = sqrtf(nrm);
    return nrm;
}

void normalize_vector(int dimension, float* vector) {
    float nrm = get_l2(dimension, vector);
    if (nrm <= s_epsinon) {
        vector[0] = 1.0;
        return;
    }
    float dnrm = 1.0 / nrm;
    for (int idx = 0; idx < dimension; ++idx) {
        vector[idx] *= dnrm;
    }
    return;
}
#endif

void normalize_matrix(int dimension, int rows, float* matrix) {
#ifdef TOOL_CALC_OPENMP
#pragma omp parallel for
#endif
    for (int idx = 0; idx < rows; ++idx) {
        normalize_vector(dimension, peek_vector(dimension, idx, matrix));
    }
    return;
}

#endif

void dis_l2_vectors(int dimension, float *v0, float *v1, float *dis) {
    float s = 0.0;
    for (int idx = 0; idx < dimension; ++idx) {
        float d = *v1 - *v0;
        ++v0;
        ++v1;
        s += d * d;
    }

    if (s <= s_epsinon) {
        *dis = 0.0;
        return;
    }
    
    *dis = sqrtf(s);
    return;
}

void dis_l2_matrix_vector(int dimension, int rows, float *matrix, float *vector, float *dises) {
    for (int idx = 0; idx < rows; ++idx) {
        dis_l2_vectors(dimension, peek_vector(dimension, idx, matrix), vector, dises + idx);
    }
    return;
}

#ifdef TOOL_CALC_MKL
void dot_vectors(int dimension, float *v0, float *v1, float *dot) {
    int incx = 1;
    *dot = cblas_sdot(dimension, v0, incx, v1, incx);
}

void dot_matrix_matrix(int dimension, int a_rows, int b_rows, float* a_matrix, float* b_matrix, float* dots) {
    CBLAS_LAYOUT  layout = CblasRowMajor;
    CBLAS_TRANSPOSE transa = CblasNoTrans;
    CBLAS_TRANSPOSE transb = CblasTrans;
    float alpha = 1.0;
    float beta = 0.0;
    cblas_sgemm(layout, transa, transb, a_rows, b_rows, dimension, alpha, 
        a_matrix, dimension, b_matrix, dimension,
        beta, dots, b_rows
    );
}

void dot_matrix_matrix_int8(int dimension, int a_rows, int b_rows, int8_t* a_matrix, int8_t* b_matrix, int32_t* dots) {
    char transa = 'N';
    char transb = 'T';
    char offsetc = 'F';
    float alpha = 1.0f;
    float beta = 0.0f;
    int8_t ao = 0;
    int8_t bo = 0;
    int32_t co = 0;
 
    dnnl_gemm_s8s8s32(transa, transb, offsetc, a_rows, 
                b_rows, dimension, alpha, a_matrix, 
                dimension, ao, b_matrix, dimension, bo,
                beta, dots, b_rows, &co);
}

void dot_matrix_vector(int dimension, int rows, float *matrix, float *vector, float *dots) {
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_TRANSPOSE trans = CblasNoTrans;
    float alpha = 1.0;
    float beta = 0.0;
    int incx = 1;
    cblas_sgemv(layout, trans, rows, dimension, alpha, matrix, dimension, vector, incx, beta, 
        dots, incx);
}

void dis_l2_matrix_vector_buf(int dimension, int rows, float *matrix, float *vector, float *dises, float* buffer) {
    int incx = 1;
    for (int row = 0; row < rows; ++row) {
        vsSub(dimension, matrix, vector, buffer);
        matrix += dimension;
        float nrm = cblas_snrm2(dimension, buffer, incx);
        if (nrm <= s_epsinon) {
            *(dises + row) = 0.0;
        } else {
            *(dises + row) = nrm;
        }
    }
}

void put_vector(int dimension, int rowid, float* matrix, float* vector) {
    int incx = 1;
    cblas_scopy(dimension, vector, incx, matrix + dimension * rowid, incx);
}

void put_vector_int8(int dimension, int rowid, int8_t* matrix, int8_t* vector) { 
    // int incx = 1;
    // cblas_ccopy(dimension, vector, incx, matrix + dimension * rowid, incx);
    memcpy(peek_vector_int8(dimension, rowid, matrix), vector, dimension * sizeof(int8_t));
}

void get_vector(int dimension, int rowid, float* matrix, float* vector) {
    int incx = 1;
    cblas_scopy(dimension, matrix + dimension * rowid, incx, vector, incx);
}

void get_vector_int8(int dimension, int rowid, int8_t* matrix, int8_t* vector) {
    // int incx = 1;
    // cblas_ccopy(dimension, matrix + dimension * rowid, incx, vector, incx);
    memcpy(vector, peek_vector_int8(dimension, rowid, matrix), dimension * sizeof(int8_t));
}
        
void axpy_vector(int dimension, float scalar, float* vector_x, float* vector_y) {
    int inc = 1;
    cblas_saxpy(dimension, scalar, vector_x, inc, vector_y, inc);
}

void product_vector(int dimension, float scalar, float* vector) {
    int inc = 1;
    cblas_sscal(dimension, scalar, vector, inc);
}

/* need to sgemv */
void product_center(int dimension, int rows, float *matrix, int scalar, float* center) {
    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_TRANSPOSE trans = CblasTrans;
    float *vector = alloc_vector(rows);
    for (int i = 0; i < rows; ++i) {
        vector[i] = 1.0;
    }
    float alpha = 1.0;
    float beta = scalar;
    int incx = 1;
    cblas_sgemv(layout, trans, rows, dimension, alpha, matrix, dimension, vector, incx, beta, 
        center, incx);
    free_vector(vector);
    normalize_vector(dimension, center);
}

#else
#ifdef TOOL_CALC_ARM
// TODO: dimension % 4 != 0
void dot_vectors(int dimension, float *v0, float *v1, float *dot) {
    float sum = 0.0;
    float32x4_t x4_sum = vdupq_n_f32(0.0f);
    for (int idx = 0; idx < dimension; idx += 4) {
        // x4_sum = x4_v0[idx] * x4_v1[idx] + x4_sum   or vfmaq_f32
        x4_sum = vmlaq_f32(x4_sum, vld1q_f32(v0 + idx), vld1q_f32(v1 + idx));
    }
    // TODO: *sum = vaddvq_f32(x4_sum) only surport a64 
    float32x2_t _ss = vadd_f32(vget_high_f32(x4_sum), vget_low_f32(x4_sum));
    sum += vget_lane_f32(vpadd_f32(_ss, _ss), 0);
    *dot = sum;
}

// TODO: dimension % 4 != 0
void axpy_vector(int dimension, float scalar, float* vector_x, float* vector_y) {
    float32x4_t _scalar = vdupq_n_f32(scalar);
    for (int idx = 0; idx < dimension; idx += 4) {
        vst1q_f32(vector_y + idx, vmlaq_f32(vld1q_f32(vector_y + idx), _scalar, vld1q_f32(vector_x + idx)));
    }
}

#else
void dot_vectors(int dimension, float *v0, float *v1, float *dot) {
    float sum = 0.0;
    for (int idx = 0; idx < dimension; ++idx) {
        sum += v0[idx] * v1[idx];
    }
    *dot = sum;
}

void axpy_vector(int dimension, float scalar, float* vector_x, float* vector_y) {
    for (int idx = 0; idx < dimension; ++idx) {
        vector_y[idx] += scalar * vector_x[idx];
    }
}

#endif

void dot_matrix_vector(int dimension, int rows, float *matrix, float *vector, float *dots) {
#ifdef TOOL_CALC_OPENMP
#pragma omp parallel for
#endif
    for (int idx = 0; idx < rows; ++idx) {
        dot_vectors(dimension, peek_vector(dimension, idx, matrix), vector, dots + idx);
    }
}

void dis_l2_matrix_vector_buf(int dimension, int rows, float *matrix, float *vector, float *dises, float* buffer) {
    (void)buffer;
    dis_l2_matrix_vector(dimension, rows, matrix, vector, dises);
}

void put_vector(int dimension, int rowid, float* matrix, float* vector) {
    memcpy(peek_vector(dimension, rowid, matrix), vector, dimension * sizeof(float));
}

void get_vector(int dimension, int rowid, float* matrix, float* vector) {
    memcpy(vector, peek_vector(dimension, rowid, matrix), dimension * sizeof(float));
}

void product_vector(int dimension, float scalar, float* vector) {
    for (int idx = 0; idx < dimension; ++idx) {
        vector[idx] *= scalar;
    }
}

void product_center(int dimension, int rows, float *matrix, int scalar, float* center) {
    for (int dim = 0; dim < dimension; ++dim) {
        float* ptr = matrix + dim;
        float sum = center[dim] * scalar;
        for (int idx = 0; idx < rows; ++idx) {
            sum += *ptr;
            ptr += dimension;
        }
        center[dim] = sum;
    }
    normalize_vector(dimension, center);
}

#endif

} // namespace calc
} // namespace ann
} // namespace tool

