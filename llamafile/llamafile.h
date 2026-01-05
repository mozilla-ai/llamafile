#ifndef LLAMAFILE_H_
#define LLAMAFILE_H_
#include <stdbool.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// FLAGS - Global configuration variables
// Most are defined in llamafile.c but only a few are actively used:
//   USED: FLAG_nologo (chatbot_main.cpp), FLAG_log_disable (chatbot_comm.cpp),
//         FLAG_gpu (internal, llamafile.c)
//   UNUSED: All others - kept for potential future use or server integration
// =============================================================================

extern bool FLAGS_READY;                  // UNUSED: not referenced
extern bool FLAG_ascii;                   // UNUSED: not referenced
extern bool FLAG_completion_mode;         // UNUSED: not referenced
extern bool FLAG_fast;                    // UNUSED: not referenced
extern bool FLAG_iq;                      // UNUSED: not referenced
extern bool FLAG_log_disable;             // USED: chatbot_comm.cpp
extern bool FLAG_mlock;                   // UNUSED: not referenced
extern bool FLAG_mmap;                    // UNUSED: not referenced
extern bool FLAG_no_display_prompt;       // UNUSED: not referenced
extern bool FLAG_nocompile;               // UNUSED: not referenced
extern bool FLAG_nologo;                  // USED: chatbot_main.cpp
extern bool FLAG_precise;                 // UNUSED: not referenced
extern bool FLAG_recompile;               // UNUSED: not referenced
extern bool FLAG_tinyblas;                // UNUSED: not referenced
extern bool FLAG_trace;                   // UNUSED: not referenced
extern bool FLAG_trap;                    // UNUSED: not referenced
extern bool FLAG_unsecure;                // UNUSED: not referenced
extern bool FLAG_v2;                      // UNUSED: not referenced
extern const char *FLAG_chat_template;    // UNUSED: not referenced
extern const char *FLAG_db;               // UNUSED: not referenced
extern const char *FLAG_db_startup_sql;   // UNUSED: not referenced
extern const char *FLAG_file;             // UNUSED: not referenced
extern const char *FLAG_ip_header;        // UNUSED: not referenced
extern const char *FLAG_listen;           // UNUSED: not referenced
extern const char *FLAG_mmproj;           // UNUSED: not referenced
extern const char *FLAG_model;            // UNUSED: not referenced
extern const char *FLAG_prompt;           // UNUSED: not referenced
extern const char *FLAG_url_prefix;       // UNUSED: not referenced
extern const char *FLAG_www_root;         // UNUSED: not referenced
extern double FLAG_token_rate;            // UNUSED: not referenced
extern float FLAG_decay_growth;           // UNUSED: not referenced
extern float FLAG_frequency_penalty;      // UNUSED: not referenced
extern float FLAG_presence_penalty;       // UNUSED: not referenced
extern float FLAG_reserve_tokens;         // UNUSED: not referenced
extern float FLAG_temperature;            // UNUSED: not referenced
extern float FLAG_top_p;                  // UNUSED: not referenced
extern int FLAG_batch;                    // UNUSED: not referenced
extern int FLAG_ctx_size;                 // UNUSED: not referenced
extern int FLAG_decay_delay;              // UNUSED: not referenced
extern int FLAG_flash_attn;               // UNUSED: not referenced
extern int FLAG_gpu;                      // USED: internal (llamafile.c)
extern int FLAG_http_ibuf_size;           // UNUSED: not referenced
extern int FLAG_http_obuf_size;           // UNUSED: not referenced
extern int FLAG_keepalive;                // UNUSED: not referenced
extern int FLAG_main_gpu;                 // UNUSED: not referenced
extern int FLAG_n_gpu_layers;             // UNUSED: not referenced
extern int FLAG_slots;                    // UNUSED: not referenced
extern int FLAG_split_mode;               // UNUSED: not referenced
extern int FLAG_threads;                  // UNUSED: not referenced
extern int FLAG_threads_batch;            // UNUSED: not referenced
extern int FLAG_token_burst;              // UNUSED: not referenced
extern int FLAG_token_cidr;               // UNUSED: not referenced
extern int FLAG_ubatch;                   // UNUSED: not referenced
extern int FLAG_verbose;                  // UNUSED: not referenced
extern int FLAG_warmup;                   // UNUSED: not referenced
extern int FLAG_workers;                  // UNUSED: not referenced
extern unsigned FLAG_seed;                // UNUSED: not referenced

// =============================================================================
// File I/O - GGUF file handling with zip support
// Defined in llamafile.c, used internally for model loading
// UNUSED externally: These are defined but not called from outside llamafile.c
// =============================================================================

struct llamafile;
struct llamafile *llamafile_open_gguf(const char *, const char *);  // UNUSED externally
void llamafile_close(struct llamafile *);                           // UNUSED externally
long llamafile_read(struct llamafile *, void *, size_t);            // UNUSED externally
long llamafile_write(struct llamafile *, const void *, size_t);     // UNUSED externally
bool llamafile_seek(struct llamafile *, size_t, int);               // UNUSED externally
void *llamafile_content(struct llamafile *);                        // UNUSED externally
size_t llamafile_tell(struct llamafile *);                          // UNUSED externally
size_t llamafile_size(struct llamafile *);                          // UNUSED externally
size_t llamafile_position(struct llamafile *);                      // UNUSED externally
bool llamafile_eof(struct llamafile *file);                         // UNUSED externally
FILE *llamafile_fp(struct llamafile *);                             // UNUSED externally
void llamafile_ref(struct llamafile *);                             // UNUSED externally
void llamafile_unref(struct llamafile *);                           // UNUSED externally

// =============================================================================
// Utility functions
// =============================================================================

// NOT DEFINED: Declaration only, no implementation in llamafile_new/
void llamafile_govern(void);                              // NOT DEFINED
void llamafile_check_cpu(void);                           // NOT DEFINED
void llamafile_help(const char *);                        // NOT DEFINED
void llamafile_log_command(char *[]);                     // NOT DEFINED
const char *llamafile_get_tmp_dir(void);                  // NOT DEFINED
void llamafile_schlep(const void *, size_t);              // NOT DEFINED
void llamafile_launch_browser(const char *);              // NOT DEFINED
void llamafile_get_flags(int, char **);                   // NOT DEFINED
char *llamafile_get_prompt(void);                         // NOT DEFINED

// USED: Defined in llamafile.c
bool llamafile_has(char **, const char *);
void llamafile_get_app_dir(char *, size_t);
bool llamafile_extract(const char *, const char *);
int llamafile_is_file_newer_than(const char *, const char *);

// =============================================================================
// GPU detection and configuration
// =============================================================================

#define LLAMAFILE_GPU_ERROR -2
#define LLAMAFILE_GPU_DISABLE -1
#define LLAMAFILE_GPU_AUTO 0
#define LLAMAFILE_GPU_AMD 1
#define LLAMAFILE_GPU_APPLE 2
#define LLAMAFILE_GPU_NVIDIA 4

bool llamafile_has_gpu(void);             // Defined in llamafile.c
bool llamafile_has_metal(void);           // Defined in metal.c (dynamic loader)
bool llamafile_has_cuda(void);            // Defined in llamafile.c (stub)
bool llamafile_has_amd_gpu(void);         // Defined in llamafile.c (stub)
int llamafile_gpu_layers(int);            // Defined in llamafile.c
int llamafile_gpu_parse(const char *);    // Defined in llamafile.c
const char *llamafile_describe_gpu(void); // Defined in llamafile.c

// Log callback type for Metal backend (matches ggml_log_callback)
typedef void (*llamafile_log_callback)(int level, const char *text, void *user_data);

// No-op log callback to disable logging (defined in llamafile.c)
void llamafile_log_callback_null(int level, const char *text, void *user_data);

// Set logging callback for Metal dylib (defined in metal.c)
// Pass a no-op callback to disable logging
void llamafile_metal_log_set(llamafile_log_callback log_callback, void *user_data);

#ifdef __cplusplus
}
#endif
#endif /* LLAMAFILE_H_ */
