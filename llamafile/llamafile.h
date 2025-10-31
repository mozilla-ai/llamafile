#ifndef LLAMAFILE_H_
#define LLAMAFILE_H_
#include <stdbool.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif

extern bool FLAGS_READY;
extern bool FLAG_ascii;
extern bool FLAG_completion_mode;
extern bool FLAG_fast;
extern bool FLAG_iq;
extern bool FLAG_log_disable;
extern bool FLAG_mlock;
extern bool FLAG_mmap;
extern bool FLAG_no_display_prompt;
extern bool FLAG_nocompile;
extern bool FLAG_nologo;
extern bool FLAG_precise;
extern bool FLAG_recompile;
extern bool FLAG_tinyblas;
extern bool FLAG_trace;
extern bool FLAG_trap;
extern bool FLAG_unsecure;
extern bool FLAG_v2;
extern const char *FLAG_chat_template;
extern const char *FLAG_db;
extern const char *FLAG_db_startup_sql;
extern const char *FLAG_file;
extern const char *FLAG_ip_header;
extern const char *FLAG_listen;
extern const char *FLAG_mmproj;
extern const char *FLAG_model;
extern const char *FLAG_prompt;
extern const char *FLAG_url_prefix;
extern const char *FLAG_www_root;
extern double FLAG_token_rate;
extern float FLAG_decay_growth;
extern float FLAG_frequency_penalty;
extern float FLAG_presence_penalty;
extern float FLAG_reserve_tokens;
extern float FLAG_temperature;
extern float FLAG_top_p;
extern float FLAG_min_p;
extern int FLAG_batch;
extern int FLAG_ctx_size;
extern int FLAG_decay_delay;
extern int FLAG_flash_attn;
extern int FLAG_gpu;
extern int FLAG_top_k;
extern int FLAG_gpu;
extern int FLAG_http_ibuf_size;
extern int FLAG_http_obuf_size;
extern int FLAG_keepalive;
extern int FLAG_main_gpu;
extern int FLAG_n_gpu_layers;
extern int FLAG_slots;
extern int FLAG_split_mode;
extern int FLAG_threads;
extern int FLAG_threads_batch;
extern int FLAG_token_burst;
extern int FLAG_token_cidr;
extern int FLAG_ubatch;
extern int FLAG_verbose;
extern int FLAG_warmup;
extern int FLAG_workers;
extern unsigned FLAG_seed;

struct llamafile;
struct llamafile *llamafile_open_gguf(const char *, const char *);
void llamafile_close(struct llamafile *);
long llamafile_read(struct llamafile *, void *, size_t);
long llamafile_write(struct llamafile *, const void *, size_t);
bool llamafile_seek(struct llamafile *, size_t, int);
void *llamafile_content(struct llamafile *);
size_t llamafile_tell(struct llamafile *);
size_t llamafile_size(struct llamafile *);
size_t llamafile_position(struct llamafile *);
bool llamafile_eof(struct llamafile *file);
FILE *llamafile_fp(struct llamafile *);
void llamafile_ref(struct llamafile *);
void llamafile_unref(struct llamafile *);
char *llamafile_get_prompt(void);

void llamafile_govern(void);
void llamafile_check_cpu(void);
void llamafile_help(const char *);
void llamafile_log_command(char *[]);
const char *llamafile_get_tmp_dir(void);
bool llamafile_has(char **, const char *);
bool llamafile_extract(const char *, const char *);
int llamafile_is_file_newer_than(const char *, const char *);
void llamafile_schlep(const void *, size_t);
void llamafile_get_app_dir(char *, size_t);
void llamafile_launch_browser(const char *);
void llamafile_get_flags(int, char **);

#define LLAMAFILE_GPU_ERROR -2
#define LLAMAFILE_GPU_DISABLE -1
#define LLAMAFILE_GPU_AUTO 0
#define LLAMAFILE_GPU_AMD 1
#define LLAMAFILE_GPU_APPLE 2
#define LLAMAFILE_GPU_NVIDIA 4
bool llamafile_has_gpu(void);
int llamafile_gpu_layers(int);
bool llamafile_has_cuda(void);
bool llamafile_has_metal(void);
bool llamafile_has_amd_gpu(void);
int llamafile_gpu_parse(const char *);
const char *llamafile_describe_gpu(void);

#ifdef __cplusplus
}
#endif
#endif /* LLAMAFILE_H_ */
