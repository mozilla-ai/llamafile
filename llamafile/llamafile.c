// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2023 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "llamafile.h"
#include "zip.h"
#include <cosmo.h>
#include <libc/assert.h>
#include <libc/str/str.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define Min(a, b) ((a) < (b) ? (a) : (b))

__notice(llamafile_notice, "\
llamafile (Apache 2.0)\n\
Copyright 2023 Mozilla Foundation\n\
\n\
Licensed under the Apache License, Version 2.0 (the \"License\");\n\
you may not use this file except in compliance with the License.\n\
You may obtain a copy of the License at\n\
\n\
    http://www.apache.org/licenses/LICENSE-2.0\n\
\n\
Unless required by applicable law or agreed to in writing, software\n\
distributed under the License is distributed on an \"AS IS\" BASIS,\n\
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n\
See the License for the specific language governing permissions and\n\
limitations under the License.\"");

struct llamafile {
    FILE *fp;
    size_t size;
    char *content;
    size_t position;
    void *mapping;
    size_t mapsize;
    char fname[PATH_MAX];
    atomic_int refs;
};

static struct llamafile *llamafile_open_zip(const char *prog, const char *fname, const char *mode) {
    int fd = -1;
    uint8_t *bufdata = NULL;
    size_t cdirsize = 0;
    uint8_t *cdirdata = NULL;
    struct llamafile *file = NULL;

    if (!(file = calloc(1, sizeof(struct llamafile))))
        return 0;
    strlcpy(file->fname, prog, PATH_MAX);

    // try opening from this executable's zip store
    if ((fd = open(prog, O_RDONLY | O_CLOEXEC)) == -1) {
        free(file);
        return 0;
    }
    ssize_t rc;
    if ((rc = lseek(fd, 0, SEEK_END)) == -1)
        goto Failure;
    file->size = rc;

    // read the last 64kb of file
    // the zip file format magic can be anywhere in there
    int amt;
    uint64_t off;
    if (file->size <= 65536) {
        off = 0;
        amt = file->size;
    } else {
        off = file->size - 65536;
        amt = file->size - off;
    }
    if (!(bufdata = gc(malloc(65536))))
        goto Failure;
    if (pread(fd, bufdata, amt, off) != amt) {
        fprintf(stderr, "%s: warning: failed to read last 64kb of file: %s\n", prog,
                strerror(errno));
        goto Failure;
    }

    // search backwards for the end-of-central-directory record
    // the eocd (cdir) says where the central directory (cfile) array is located
    // we consistency check some legacy fields, to be extra sure that it is eocd
    unsigned cnt = 0;
    for (int i = amt - Min(kZipCdirHdrMinSize, kZipCdir64LocatorSize); i >= 0; --i) {
        uint32_t magic = ZIP_READ32(bufdata + i);
        if (magic == kZipCdir64LocatorMagic && i + kZipCdir64LocatorSize <= amt &&
            pread(fd, bufdata, kZipCdir64HdrMinSize, ZIP_LOCATE64_OFFSET(bufdata + i)) ==
                (long)kZipCdir64HdrMinSize &&
            ZIP_READ32(bufdata) == kZipCdir64HdrMagic &&
            ZIP_CDIR64_RECORDS(bufdata) == ZIP_CDIR64_RECORDSONDISK(bufdata) &&
            ZIP_CDIR64_RECORDS(bufdata) && ZIP_CDIR64_SIZE(bufdata) <= INT_MAX) {
            cnt = ZIP_CDIR64_RECORDS(bufdata);
            off = ZIP_CDIR64_OFFSET(bufdata);
            amt = ZIP_CDIR64_SIZE(bufdata);
            break;
        }
        if (magic == kZipCdirHdrMagic && i + kZipCdirHdrMinSize <= amt &&
            ZIP_CDIR_RECORDS(bufdata + i) == ZIP_CDIR_RECORDSONDISK(bufdata + i) &&
            ZIP_CDIR_RECORDS(bufdata + i) && ZIP_CDIR_SIZE(bufdata + i) <= INT_MAX &&
            ZIP_CDIR_OFFSET(bufdata + i) != 0xffffffffu) {
            cnt = ZIP_CDIR_RECORDS(bufdata + i);
            off = ZIP_CDIR_OFFSET(bufdata + i);
            amt = ZIP_CDIR_SIZE(bufdata + i);
            break;
        }
    }
    if (cnt <= 0) {
        // this executable isn't a zip file
        fprintf(stderr, "%s: warning: not a pkzip archive\n", prog);
        goto Invalid;
    }

    // read the central directory
    cdirsize = amt;
    if (!(cdirdata = gc(malloc(cdirsize))))
        goto Failure;
    if (pread(fd, cdirdata, cdirsize, off) != (long)cdirsize) {
        fprintf(stderr, "%s: warning: failed to pread zip cdir: %s\n", prog, strerror(errno));
        goto Failure;
    }
    if (ZIP_READ32(cdirdata) != kZipCfileHdrMagic) {
        fprintf(stderr, "%s: warning: failed to locate zip central directory\n", prog);
        goto Invalid;
    }

    // look for filename in the directory
    int found = 0;
    char *zip_name = 0;
    unsigned cdir_offset;
    int fname_len = fname ? strlen(fname) : 0;
    unsigned entry_index, entry_offset;
    for (entry_index = entry_offset = 0;
         entry_index < cnt && entry_offset + kZipCfileHdrMinSize <= cdirsize &&
         entry_offset + ZIP_CFILE_HDRSIZE(cdirdata + entry_offset) <= cdirsize;
         ++entry_index, entry_offset += ZIP_CFILE_HDRSIZE(cdirdata + entry_offset)) {
        if (ZIP_CFILE_MAGIC(cdirdata + entry_offset) != kZipCfileHdrMagic) {
            fprintf(stderr, "error: corrupted zip central directory entry magic: %s\n", prog);
            errno = EINVAL;
            goto Failure;
        }
        int entry_name_len = ZIP_CFILE_NAMESIZE(cdirdata + entry_offset);
        const char *entry_name_bytes = ZIP_CFILE_NAME(cdirdata + entry_offset);
        if ((fname ? (fname_len == entry_name_len && !memcmp(fname, entry_name_bytes, fname_len))
                   : (entry_name_len > 5 &&
                      !memcasecmp(entry_name_bytes + entry_name_len - 5, ".gguf", 5)))) {
            zip_name = gc(strndup(entry_name_bytes, entry_name_len));
            off = get_zip_cfile_offset(cdirdata + entry_offset);
            file->size = get_zip_cfile_compressed_size(cdirdata + entry_offset);
            cdir_offset = entry_offset;
            ++found;
        }
    }
    if (!found) {
        fprintf(stderr, "%s: error: no %s file found in zip archive\n", prog,
                fname ? fname : ".gguf");
        goto Invalid;
    }
    if (found != 1) {
        // TODO: Support opening LLaVA llamafiles.
        fprintf(stderr, "%s: error: multiple %s files found in zip archive\n", prog,
                fname ? fname : ".gguf");
        goto Invalid;
    }
    strlcat(file->fname, "@", PATH_MAX);
    strlcat(file->fname, zip_name, PATH_MAX);
    if (ZIP_CFILE_COMPRESSIONMETHOD(cdirdata + cdir_offset) != kZipCompressionNone) {
        fprintf(
            stderr,
            "%s: error: weights stored in the zip executable can't be stored using compression\n",
            file->fname);
        goto Invalid;
    }

    // read the zip local file header
    // this is needed to determine offset of file content
    uint8_t lfile[kZipLfileHdrMinSize];
    if (pread(fd, lfile, kZipLfileHdrMinSize, off) != kZipLfileHdrMinSize) {
        fprintf(stderr, "%s: error: failed to pread lfile\n", file->fname);
        goto Failure;
    }
    if (ZIP_LFILE_MAGIC(lfile) != kZipLfileHdrMagic) {
        fprintf(stderr, "%s: error: corrupted zip local file magic\n", file->fname);
        goto Invalid;
    }
    off += ZIP_LFILE_HDRSIZE(lfile);

    // perform sanity check
    // mapping weights for apple metal gpu requires 16kb alignment
    if (off & 16383)
        fprintf(stderr, "%s: warning: use zipalign (rather than zip) to create llamafiles\n",
                file->fname);

    // map the file into memory
    long pagesz = sysconf(_SC_GRANSIZE);
    off_t mapoff = off & -pagesz;
    long skew = off - mapoff;
    file->mapsize = skew + file->size;
    file->mapping = mmap(0, file->mapsize, PROT_READ, MAP_SHARED, fd, mapoff);
    if (file->mapping == MAP_FAILED) {
        fprintf(stderr, "%s: warning: failed to map zip file: %s\n", file->fname, strerror(errno));
        goto Failure;
    }

    errno_t err;
    if ((err = posix_fadvise(fd, mapoff, file->mapsize, POSIX_FADV_SEQUENTIAL)) && err != ENOSYS)
        fprintf(stderr, "%s: warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n",
                file->fname, strerror(err));

    // setup our synthetic file
    file->position = 0;
    file->content = (char *)file->mapping + skew;

    // return object
    close(fd);
    return file;

Invalid:
    errno = EINVAL;
Failure:
    free(file);
    close(fd);
    return 0;
}

static struct llamafile *llamafile_open_file(const char *fname, const char *mode) {
    struct llamafile *file;
    if (!(file = calloc(1, sizeof(struct llamafile))))
        return 0;
    strlcpy(file->fname, fname, PATH_MAX);
    if ((file->fp = fopen(fname, mode))) {
        if (!llamafile_seek(file, 0, SEEK_END)) {
            llamafile_close(file);
            return 0;
        }
        file->size = llamafile_tell(file);
        llamafile_seek(file, 0, SEEK_SET);
        return file;
    }
    free(file);
    return 0;
}

struct llamafile *llamafile_open_gguf(const char *fname, const char *mode) {

    // support filenames like `foo.zip@weights.gguf`
    const char *p;
    if ((p = strchr(fname, '@')))
        return llamafile_open_zip(gc(strndup(fname, p - fname)), p + 1, mode);

    // open from file or from our own executable if it doesn't exist
    struct llamafile *file;
    if (!(file = llamafile_open_file(fname, mode))) {
        if (errno == ENOENT) {
            if (!(file = llamafile_open_zip(GetProgramExecutableName(), fname, mode))) {
                errno = ENOENT;
                return 0;
            }
            return file;
        } else {
            return 0;
        }
    }

    // check that this is a .gguf file
    ssize_t rc;
    char buf[8];
    if ((rc = pread(fileno(file->fp), buf, 8, 0)) == -1) {
        llamafile_close(file);
        return 0;
    }
    if (rc != 8) {
        llamafile_close(file);
        errno = EIO;
        return 0;
    }
    if (ZIP_READ32(buf) == ZIP_READ32("GGUF") || ZIP_READ32(buf) == ZIP_READ32("ggml")) {
        errno = EINVAL;
        return file;
    }

    // otherwise assume user opened a .zip or .llamafile
    llamafile_close(file);
    return llamafile_open_zip(fname, 0, mode);
}

FILE *llamafile_fp(struct llamafile *file) {
    return file->fp;
}

size_t llamafile_size(struct llamafile *file) {
    return file->size;
}

size_t llamafile_position(struct llamafile *file) {
    return file->position;
}

bool llamafile_eof(struct llamafile *file) {
    if (file->fp)
        return feof(file->fp);
    return file->position >= file->size;
}

void *llamafile_content(struct llamafile *file) {
    return file->content;
}

size_t llamafile_tell(struct llamafile *file) {
    if (!file->fp)
        return file->position;
    long ret = ftell(file->fp);
    assert(ret != -1); // shouldn't fail because we seeked earlier
    return (size_t)ret;
}

bool llamafile_seek(struct llamafile *file, size_t offset, int whence) {
    if (!file->fp) {
        switch (whence) {
        case SEEK_SET:
            file->position = offset;
            break;
        case SEEK_CUR:
            file->position += offset;
            break;
        case SEEK_END:
            file->position = file->size + offset;
            break;
        }
        return true;
    }
    return !fseek(file->fp, (long)offset, whence);
}

long llamafile_read(struct llamafile *file, void *ptr, size_t len) {
    if (len == 0)
        return 0;
    if (!file->fp) {
        if (file->position > file->size)
            return 0;
        size_t remain = file->size - file->position;
        size_t amt = Min(len, remain);
        memcpy(ptr, file->content + file->position, amt);
        file->position += amt;
        return amt;
    }
    errno = 0;
    size_t ret = fread(ptr, len, 1, file->fp);
    if (ferror(file->fp))
        return -1;
    if (ret != 1)
        return 0;
    return len;
}

long llamafile_write(struct llamafile *file, const void *ptr, size_t len) {
    if (len == 0)
        return 0;
    if (!file->fp) {
        errno = EROFS;
        return -1;
    }
    errno = 0;
    size_t ret = fwrite(ptr, len, 1, file->fp);
    if (ferror(file->fp))
        return -1;
    if (ret != 1)
        return 0;
    return len;
}

static void llamafile_close_impl(struct llamafile *file) {
    if (file->fp)
        fclose(file->fp);
    if (file->mapping && file->mapping != MAP_FAILED) {
        munmap(file->mapping, file->mapsize);
    }
    free(file);
}

void llamafile_ref(struct llamafile *file) {
    atomic_fetch_add(&file->refs, 1);
}

void llamafile_unref(struct llamafile *file) {
    if (!atomic_fetch_sub(&file->refs, 1)) {
        llamafile_close_impl(file);
    }
}

void llamafile_close(struct llamafile *file) {
    llamafile_unref(file);
}

// ==============================================================================
// FLAG variable definitions
// ==============================================================================
// These are Cosmopolitan FLAG system variables. For migration to other
// platforms, replace these with your own configuration system.

bool FLAGS_READY = false;
bool FLAG_ascii = false;
bool FLAG_completion_mode = false;
bool FLAG_fast = false;
bool FLAG_iq = false;
bool FLAG_log_disable = false;
bool FLAG_mlock = false;
bool FLAG_mmap = true;
bool FLAG_no_display_prompt = false;
bool FLAG_nocompile = false;
bool FLAG_nologo = false;
bool FLAG_precise = false;
bool FLAG_recompile = false;
bool FLAG_tinyblas = false;
bool FLAG_trace = false;
bool FLAG_trap = false;
bool FLAG_unsecure = false;
bool FLAG_v2 = false;
const char *FLAG_chat_template = NULL;
const char *FLAG_db = NULL;
const char *FLAG_db_startup_sql = NULL;
const char *FLAG_file = NULL;
const char *FLAG_ip_header = NULL;
const char *FLAG_listen = "127.0.0.1:8080";
const char *FLAG_mmproj = NULL;
const char *FLAG_model = NULL;
const char *FLAG_prompt = NULL;
const char *FLAG_url_prefix = "";
const char *FLAG_www_root = NULL;
double FLAG_token_rate = 0;
float FLAG_decay_growth = 0;
float FLAG_frequency_penalty = 0;
float FLAG_presence_penalty = 0;
float FLAG_reserve_tokens = 0;
float FLAG_temperature = 0.8;
float FLAG_top_p = 0.95;
int FLAG_batch = 512;
int FLAG_ctx_size = 0;
int FLAG_decay_delay = 0;
int FLAG_flash_attn = 0;
int FLAG_gpu = LLAMAFILE_GPU_AUTO;
int FLAG_http_ibuf_size = 65536;
int FLAG_http_obuf_size = 65536;
int FLAG_keepalive = 5;
int FLAG_main_gpu = 0;
int FLAG_n_gpu_layers = -1;
int FLAG_slots = 1;
int FLAG_split_mode = 0;
int FLAG_threads = 0;
int FLAG_threads_batch = 0;
int FLAG_token_burst = 0;
int FLAG_token_cidr = 0;
int FLAG_ubatch = 512;
int FLAG_verbose = 0;
int FLAG_warmup = 0;
int FLAG_workers = 0;
unsigned FLAG_seed = 0;

// ==============================================================================
// Utility functions
// ==============================================================================

bool llamafile_has(char **a, const char *x) {
    for (int i = 0; a[i]; ++i)
        if (!strcmp(a[i], x))
            return true;
    return false;
}

static const char *llamafile_get_home_dir(void) {
    const char *homedir;
    if (!(homedir = getenv("HOME")) || !*homedir)
        homedir = ".";
    return homedir;
}

/**
 * Returns path of directory for app-specific files.
 * Path includes version number: ~/.llamafile/v/<major>.<minor>.<patch>/
 * This ensures different versions don't overwrite each other's compiled dylibs.
 */
void llamafile_get_app_dir(char *path, size_t size) {
    snprintf(path, size, "%s/.llamafile/v/%d.%d.%d/",
             llamafile_get_home_dir(),
             LLAMAFILE_VERSION_MAJOR,
             LLAMAFILE_VERSION_MINOR,
             LLAMAFILE_VERSION_PATCH);
}

static int copy_file_contents(int fdin, int fdout) {
    char buf[8192];
    ssize_t nread;
    while ((nread = read(fdin, buf, sizeof(buf))) > 0) {
        char *ptr = buf;
        while (nread > 0) {
            ssize_t nwritten = write(fdout, ptr, nread);
            if (nwritten < 0) return -1;
            nread -= nwritten;
            ptr += nwritten;
        }
    }
    return nread < 0 ? -1 : 0;
}

/**
 * Returns true if `zip` was successfully copied to `to`.
 *
 * Copying happens atomically. The `zip` argument is a file system path,
 * which may reside under `/zip/...` to relocate a compressed executable
 * asset to the local filesystem.
 */
bool llamafile_extract(const char *zip, const char *to) {
    int fdin, fdout;
    char stage[PATH_MAX];
    if (FLAG_verbose)
        fprintf(stderr, "extracting %s to %s\n", zip, to);
    strlcpy(stage, to, sizeof(stage));
    if (strlcat(stage, ".XXXXXX", sizeof(stage)) >= sizeof(stage)) {
        errno = ENAMETOOLONG;
        perror(to);
        return false;
    }
    if ((fdout = mkstemp(stage)) == -1) {
        perror(stage);
        return false;
    }
    if ((fdin = open(zip, O_RDONLY | O_CLOEXEC)) == -1) {
        perror(zip);
        close(fdout);
        unlink(stage);
        return false;
    }
    if (copy_file_contents(fdin, fdout) == -1) {
        perror(zip);
        close(fdin);
        close(fdout);
        unlink(stage);
        return false;
    }
    if (close(fdout)) {
        perror(to);
        close(fdin);
        unlink(stage);
        return false;
    }
    if (close(fdin)) {
        perror(zip);
        unlink(stage);
        return false;
    }
    if (rename(stage, to)) {
        perror(to);
        unlink(stage);
        return false;
    }
    return true;
}

static int is_file_newer_than_time(const char *path, const char *other) {
    struct stat st1, st2;
    if (stat(path, &st1)) {
        perror(path);
        return -1;
    }
    if (stat(other, &st2)) {
        if (errno == ENOENT) {
            return true;
        } else {
            perror(other);
            return -1;
        }
    }
    return timespec_cmp(st1.st_mtim, st2.st_mtim) > 0;
}

static int is_file_newer_than_bytes(const char *path, const char *other) {
    int other_fd;
    if ((other_fd = open(other, O_RDONLY | O_CLOEXEC)) == -1) {
        if (errno == ENOENT) {
            return true;
        } else {
            perror(other);
            return -1;
        }
    }
    int path_fd;
    if ((path_fd = open(path, O_RDONLY | O_CLOEXEC)) == -1) {
        perror(path);
        close(other_fd);
        return -1;
    }
    int res;
    off_t i = 0;
    for (;;) {
        char path_buf[512];
        ssize_t path_rc = pread(path_fd, path_buf, sizeof(path_buf), i);
        if (path_rc == -1) {
            perror(path);
            res = -1;
            break;
        }
        char other_buf[512];
        ssize_t other_rc = pread(other_fd, other_buf, sizeof(other_buf), i);
        if (other_rc == -1) {
            perror(other);
            res = -1;
            break;
        }
        if (!path_rc || !other_rc) {
            if (!path_rc && !other_rc)
                res = false;
            else
                res = true;
            break;
        }
        size_t size = path_rc;
        if (other_rc < path_rc)
            size = other_rc;
        if (memcmp(path_buf, other_buf, size)) {
            res = true;
            break;
        }
        i += size;
    }
    if (close(path_fd)) {
        perror(path);
        res = -1;
    }
    if (close(other_fd)) {
        perror(other);
        res = -1;
    }
    return res;
}

/**
 * Returns 1 if `path` should replace `other`, 0 if not, -1 on error.
 *
 * For /zip/ paths, compares file contents byte-by-byte.
 * For regular paths, compares modification timestamps.
 */
int llamafile_is_file_newer_than(const char *path, const char *other) {
    if (startswith(path, "/zip/"))
        return is_file_newer_than_bytes(path, other);
    else
        return is_file_newer_than_time(path, other);
}

// ==============================================================================
// Logging
// ==============================================================================

void llamafile_log_callback_null(int level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

// ==============================================================================
// GPU support
// ==============================================================================
// llamafile_has_metal() is defined in metal.c with full dynamic loading support

bool llamafile_has_cuda(void) {
    // TODO: Implement CUDA/ROCm dynamic loading support
    return false;
}

bool llamafile_has_amd_gpu(void) {
    // TODO: Implement AMD GPU support
    return false;
}

bool llamafile_has_gpu(void) {
    return llamafile_has_metal() || llamafile_has_cuda();
}

const char *llamafile_describe_gpu(void) {
    switch (FLAG_gpu) {
    case LLAMAFILE_GPU_AUTO:
        return "auto";
    case LLAMAFILE_GPU_AMD:
        return "amd";
    case LLAMAFILE_GPU_APPLE:
        return "apple";
    case LLAMAFILE_GPU_NVIDIA:
        return "nvidia";
    case LLAMAFILE_GPU_DISABLE:
        return "disabled";
    default:
        return "error";
    }
}

int llamafile_gpu_parse(const char *s) {
    if (!strcasecmp(s, "disable") || !strcasecmp(s, "disabled"))
        return LLAMAFILE_GPU_DISABLE;
    if (!strcasecmp(s, "auto"))
        return LLAMAFILE_GPU_AUTO;
    if (!strcasecmp(s, "amd") || !strcasecmp(s, "rocblas") || !strcasecmp(s, "rocm") || !strcasecmp(s, "hip"))
        return LLAMAFILE_GPU_AMD;
    if (!strcasecmp(s, "apple") || !strcasecmp(s, "metal"))
        return LLAMAFILE_GPU_APPLE;
    if (!strcasecmp(s, "nvidia") || !strcasecmp(s, "cublas"))
        return LLAMAFILE_GPU_NVIDIA;
    return LLAMAFILE_GPU_ERROR;
}

int llamafile_gpu_layers(int n_gpu_layers) {
    if (FLAG_gpu == LLAMAFILE_GPU_DISABLE)
        return 0;
    if (n_gpu_layers < 0 && FLAG_gpu > 0)
        n_gpu_layers = INT_MAX;
    if (n_gpu_layers <= 0 && llamafile_has_metal())
        n_gpu_layers = INT_MAX;
    if (n_gpu_layers > 0 && !llamafile_has_gpu()) {
        FLAG_gpu = LLAMAFILE_GPU_DISABLE;
        n_gpu_layers = 0;
    }
    if (n_gpu_layers <= 0) {
        FLAG_gpu = LLAMAFILE_GPU_DISABLE;
    }
    return n_gpu_layers;
}
