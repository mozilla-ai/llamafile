// Maps the canonical <mbedtls/error.h> include path onto llamafile's
// vendored copy, for third-party code (e.g. cpp-httplib's Mbed TLS
// backend) that expects standard Mbed TLS include paths.
#include "third_party/mbedtls/error.h"
