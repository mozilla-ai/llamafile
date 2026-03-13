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

#include "worker.h"
#include "llamafile/llamafile.h"
#include "llamafile/server/client.h"
#include "llamafile/server/log.h"
#include "llamafile/server/server.h"
#include "llamafile/server/signals.h"
#include "llamafile/server/tokenbucket.h"
#include "llamafile/threadlocal.h"
#include "llamafile/trust.h"
#include <atomic>
#include <cassert>
#include <cosmo.h>
#include <exception>
#include <pthread.h>

namespace lf {
namespace server {

Worker::Worker(Server* server, llama_model* model)
  : server_(server), client_(model)
{
    dll_init(&elem_);
}

void
Worker::kill()
{
    pthread_cancel(th_);
}

void
Worker::begin()
{
    npassert(!working_);
    client_.worker_ = this;
    client_.client_ip_trusted_ = is_trusted_ip(client_.client_ip_);
    int tokens = 0;
    if (!client_.client_ip_trusted_)
        tokens = tokenbucket_acquire(client_.client_ip_);
    server_->lock();
    dll_remove(&server_->idle_workers, &elem_);
    working_ = true;
    if (tokens > FLAG_token_burst) {
        dll_make_last(&server_->active_workers, &elem_);
    } else {
        dll_make_first(&server_->active_workers, &elem_);
    }
    server_->unlock();
}

void
Worker::end()
{
    npassert(working_);
    server_->lock();
    dll_remove(&server_->active_workers, &elem_);
    working_ = false;
    dll_make_first(&server_->idle_workers, &elem_);
    server_->unlock();
}

void
Worker::deprioritize()
{
    npassert(working_);
    server_->lock();
    dll_remove(&server_->active_workers, &elem_);
    dll_make_last(&server_->active_workers, &elem_);
    server_->unlock();
}

void
Worker::retire()
{
    server_->lock();
    if (working_)
        dll_remove(&server_->active_workers, &elem_);
    else
        dll_remove(&server_->idle_workers, &elem_);
    server_->worker_count.fetch_sub(1, std::memory_order_acq_rel);
    server_->signal();
    server_->unlock();
    delete this;
}

void
Worker::handle()
{
    if ((client_.fd_ = server_->accept(&client_.client_ip_)) == -1) {
        if (IsWindows() && errno == ENOTSOCK) {
            // Server::shutdown() calls close() on the listening socket
        } else {
            SLOG("accept returned %m");
        }
        return;
    }

    begin();

    try {
        client_.run();
    } catch (const std::exception& e) {
        SLOG("caught %s", e.what());
    } catch (...) {
        SLOG("caught unknown exception");
    }

    client_.close();
    end();
}

void
Worker::run()
{
    if (!FLAG_unsecure) {
        static std::atomic<bool> once;
        if (llamafile_has_gpu()) {
            if (!once.exchange(true))
                SLOG("warning: gpu mode disables pledge security");
        } else {
            const char* promises;
            if (FLAG_www_root && !startswith(FLAG_www_root, "/zip/")) {
                promises = "stdio anet rpath";
            } else {
                promises = "stdio anet";
            }
            if (pledge(0, 0)) {
                if (!once.exchange(true))
                    SLOG("warning: this OS doesn't support pledge() security");
            } else if (pledge(promises, 0)) {
                perror("pledge");
                exit(1);
            }
        }
    }

    server_->lock();
    dll_make_first(&server_->idle_workers, &elem_);
    server_->worker_count.fetch_add(1, std::memory_order_acq_rel);
    server_->unlock();

    static ThreadLocal<Worker> cleanup([](Worker* worker) {
        if (worker->working_) {
            worker->client_.close();
            worker->end();
        }
        worker->retire();
    });
    cleanup.set(this);

    while (!server_->terminated.load(std::memory_order_acquire)) {
        sigset_t mask;
        sigemptyset(&mask);
        sigaddset(&mask, SIGHUP);
        sigaddset(&mask, SIGINT);
        sigaddset(&mask, SIGTERM);
        sigaddset(&mask, SIGUSR1);
        sigaddset(&mask, SIGALRM);
        pthread_sigmask(SIG_SETMASK, &mask, 0);
        handle();
    }

    cleanup.set(nullptr);
    retire();
}

} // namespace server
} // namespace lf
