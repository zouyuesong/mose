#ifndef QDLDL_TYPES_H
# define QDLDL_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <limits.h>

/* match OSQP's DLONG build (long long indices, double values) */
typedef long long QDLDL_int;
typedef double    QDLDL_float;
typedef signed char QDLDL_bool;

#define QDLDL_INT_MAX LLONG_MAX

#ifdef __cplusplus
}
#endif

#endif /* QDLDL_TYPES_H */
