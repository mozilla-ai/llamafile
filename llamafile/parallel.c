// -*- mode:c;indent-tabs-mode:nil;tab-width:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
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
//
// Minimal parallel job runner for Windows
//
// Reads commands from a file (one per line) and executes up to N in
// parallel using the Win32 API. Reports failures immediately and
// returns non-zero if any command failed.
//
// Usage:
//   parallel.exe [-jN] commands.txt
//   parallel.exe -j8 commands.txt
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define MAX_JOBS     64
#define MAX_CMD_LEN  8192
#define MAX_COMMANDS 4096

typedef struct {
    HANDLE process;
    DWORD  index;    // command index (for reporting)
    char   label[256];
} Job;

static int read_commands(const char *path, char **cmds, char **labels,
                         int max_cmds) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "parallel: cannot open %s\n", path);
        return -1;
    }
    int n = 0;
    char buf[MAX_CMD_LEN];
    while (n < max_cmds && fgets(buf, sizeof(buf), f)) {
        // strip trailing newline/whitespace
        int len = (int)strlen(buf);
        while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r' ||
                           buf[len-1] == ' '))
            buf[--len] = '\0';
        if (len == 0) continue;  // skip blank lines
        if (buf[0] == '#') continue;  // skip comments

        // Check for label prefix: "LABEL:command"
        char *sep = strstr(buf, ":::");
        if (sep) {
            *sep = '\0';
            labels[n] = _strdup(buf);
            cmds[n] = _strdup(sep + 3);
        } else {
            labels[n] = NULL;
            cmds[n] = _strdup(buf);
        }
        n++;
    }
    fclose(f);
    return n;
}

static HANDLE launch(const char *cmd) {
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    memset(&pi, 0, sizeof(pi));

    // CreateProcess needs a mutable copy of the command line
    char *mut_cmd = _strdup(cmd);
    if (!mut_cmd) return NULL;

    BOOL ok = CreateProcessA(
        NULL,       // application name (NULL = use command line)
        mut_cmd,    // command line
        NULL,       // process security attributes
        NULL,       // thread security attributes
        FALSE,      // inherit handles
        0,          // creation flags
        NULL,       // environment (inherit)
        NULL,       // current directory (inherit)
        &si,
        &pi
    );
    free(mut_cmd);

    if (!ok) return NULL;

    CloseHandle(pi.hThread);
    return pi.hProcess;
}

int main(int argc, char **argv) {
    int max_jobs = 4;
    const char *cmd_file = NULL;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] == 'j') {
            max_jobs = atoi(argv[i] + 2);
            if (max_jobs < 1) max_jobs = 1;
            if (max_jobs > MAX_JOBS) max_jobs = MAX_JOBS;
        } else {
            cmd_file = argv[i];
        }
    }

    if (!cmd_file) {
        fprintf(stderr, "Usage: parallel.exe [-jN] commands.txt\n");
        return 1;
    }

    // Read all commands
    char *cmds[MAX_COMMANDS];
    char *labels[MAX_COMMANDS];
    int num_cmds = read_commands(cmd_file, cmds, labels, MAX_COMMANDS);
    if (num_cmds < 0) return 1;
    if (num_cmds == 0) {
        fprintf(stderr, "parallel: no commands to run\n");
        return 0;
    }

    printf("parallel: %d commands, %d jobs\n\n", num_cmds, max_jobs);

    Job slots[MAX_JOBS];
    int num_slots = 0;
    int next_cmd = 0;
    int completed = 0;
    int failed = 0;

    while (completed < num_cmds) {
        // Fill available slots
        while (num_slots < max_jobs && next_cmd < num_cmds) {
            const char *label = labels[next_cmd] ? labels[next_cmd]
                                                 : cmds[next_cmd];
            printf("[%d/%d] %s\n", next_cmd + 1, num_cmds, label);
            fflush(stdout);

            HANDLE h = launch(cmds[next_cmd]);
            if (!h) {
                fprintf(stderr, "parallel: failed to launch: %s\n",
                        cmds[next_cmd]);
                failed++;
                completed++;
                next_cmd++;
                continue;
            }

            slots[num_slots].process = h;
            slots[num_slots].index = next_cmd;
            strncpy(slots[num_slots].label, label,
                    sizeof(slots[num_slots].label) - 1);
            slots[num_slots].label[sizeof(slots[num_slots].label) - 1] = '\0';
            num_slots++;
            next_cmd++;
        }

        if (num_slots == 0) break;

        // Build handle array for WaitForMultipleObjects
        HANDLE handles[MAX_JOBS];
        for (int i = 0; i < num_slots; i++)
            handles[i] = slots[i].process;

        DWORD result = WaitForMultipleObjects(num_slots, handles, FALSE,
                                              INFINITE);
        if (result >= WAIT_OBJECT_0 &&
            result < WAIT_OBJECT_0 + (DWORD)num_slots) {
            int idx = result - WAIT_OBJECT_0;

            DWORD exit_code = 0;
            GetExitCodeProcess(slots[idx].process, &exit_code);
            CloseHandle(slots[idx].process);

            if (exit_code != 0) {
                fprintf(stderr,
                        "\nparallel: FAILED (exit %lu): %s\n\n",
                        exit_code, slots[idx].label);
                failed++;
            }
            completed++;

            // Remove this slot by swapping with last
            if (idx < num_slots - 1)
                slots[idx] = slots[num_slots - 1];
            num_slots--;
        } else {
            fprintf(stderr, "parallel: WaitForMultipleObjects failed (%lu)\n",
                    GetLastError());
            break;
        }
    }

    // Cleanup
    for (int i = 0; i < num_cmds; i++) {
        free(cmds[i]);
        if (labels[i]) free(labels[i]);
    }

    printf("\nparallel: %d/%d succeeded", completed - failed, num_cmds);
    if (failed) printf(", %d failed", failed);
    printf("\n");

    return failed ? 1 : 0;
}
