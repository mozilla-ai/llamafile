// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
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

#include "compute.h"

#include <cosmo.h>
#include <cstring>
#include <libc/intrin/x86.h>
#include <sys/auxv.h>

#include "common.h"

static bool starts_with_str(const char *str, const char *prefix) {
    return strncmp(str, prefix, strlen(prefix)) == 0;
}

#ifdef __x86_64__
static void cpuid(unsigned leaf, unsigned subleaf, unsigned *info) {
    asm("movq\t%%rbx,%%rsi\n\t"
        "cpuid\n\t"
        "xchgq\t%%rbx,%%rsi"
        : "=a"(info[0]), "=S"(info[1]), "=c"(info[2]), "=d"(info[3])
        : "0"(leaf), "2"(subleaf));
}
#endif // __x86_64__

/**
 * Returns string describing host CPU.
 */
std::string llamafile_describe_cpu() {
    std::string id;

#ifdef __x86_64__
    union {
        char str[64];
        unsigned reg[16];
    } u = {0};
    cpuid(0x80000002, 0, u.reg + 0 * 4);
    cpuid(0x80000003, 0, u.reg + 1 * 4);
    cpuid(0x80000004, 0, u.reg + 2 * 4);
    int len = strlen(u.str);
    while (len > 0 && u.str[len - 1] == ' ')
        u.str[--len] = 0;
    id = u.str;
#else
    if (IsLinux()) {
        FILE *f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) {
                if (!strncmp(buf, "model name", 10) ||
                    starts_with_str(buf, "Model\t\t:")) { // e.g. raspi
                    char *p = strchr(buf, ':');
                    if (p) {
                        p++;
                        while (std::isspace(*p))
                            p++;
                        while (std::isspace(p[strlen(p) - 1]))
                            p[strlen(p) - 1] = '\0';
                        id = p;
                        break;
                    }
                }
            }
            fclose(f);
        }
    }
    // Note: macOS CPU detection removed for cosmopolitan cross-compilation
    // compatibility. The CPU brand string will come from /proc/cpuinfo
    // or x86 CPUID on x86_64.
#endif
    string_replace_all(id, " 96-Cores", "");
    string_replace_all(id, "(TM)", "");
    string_replace_all(id, "(R)", "");

    std::string march;
#ifdef __x86_64__
    if (__cpu_march(__cpu_model.__cpu_subtype))
        march = __cpu_march(__cpu_model.__cpu_subtype);
#else
    long hwcap = getauxval(AT_HWCAP);
    if (hwcap & HWCAP_ASIMDHP)
        march += "+fp16";
    if (hwcap & HWCAP_ASIMDDP)
        march += "+dotprod";
#endif

    if (!march.empty()) {
        bool empty = id.empty();
        if (!empty)
            id += " (";
        id += march;
        if (!empty)
            id += ")";
    }

    return id;
}
