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
#include <assert.h>
#include <cosmo.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
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
        // Multiple GGUF files found - this is OK for LLaVA models
        // Just pick the first one found and log a message
        fprintf(stderr, "%s: note: multiple GGUF files found in ZIP\n", prog);
        if (!fname) {
            fprintf(stderr, "%s: selecting '%s' (use @filename to specify)\n", prog, zip_name);
        }
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
    npassert(ret != -1); // shouldn't fail because we seeked earlier
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

const char *llamafile_name(struct llamafile *file) {
    return file ? file->fname : NULL;
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
