#ifndef OSQP_CONFIGURE_H
# define OSQP_CONFIGURE_H

# ifdef __cplusplus
extern "C" {
# endif

/* vendored build configuration for the cgo static compile */
#ifndef IS_LINUX
# define IS_LINUX
#endif
/* EMBEDDED 0: intentionally left undefined (cmake emits no define for 0) */
#ifndef PRINTING
# define PRINTING
#endif
#ifndef PROFILING
# define PROFILING
#endif
#ifndef CTRLC
# define CTRLC
#endif
#ifndef DLONG
# define DLONG
#endif

# ifdef __cplusplus
}
# endif

#endif /* ifndef OSQP_CONFIGURE_H */
