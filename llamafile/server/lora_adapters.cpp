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

#include "client.h"
#include "llama.cpp/llama.h"
#include "llama.cpp/common.h"
#include "llamafile/json.h"
#include "llamafile/llamafile.h"
#include "llamafile/server/log.h"
#include "llamafile/server/server.h"
#include "llamafile/server/worker.h"
#include "llamafile/server/slots.h"
#include "llamafile/server/slot.h"
#include <filesystem>

using jt::Json;

// External declarations for global LoRA adapter storage from prog.cpp (outside namespace)
extern std::vector<llama_lora_adapter_container> g_lora_adapters;

namespace lf {
namespace server {

bool
Client::lora_adapters()
{
    // Support both GET and POST methods
    if (msg_.method == kHttpGet) {
        // GET: Return current adapter configuration (upstream llama.cpp format)
        Json json;
        json.setArray();
        std::vector<Json>& json_array = json.getArray();
        
        for (size_t i = 0; i < ::g_lora_adapters.size(); i++) {
            Json adapter;
            adapter.setObject();
            adapter["id"] = (int)i;
            adapter["path"] = ::g_lora_adapters[i].path;
            adapter["scale"] = ::g_lora_adapters[i].scale;
            json_array.push_back(adapter);
        }
        
        char* p = append_http_response_message(obuf_.p, 200);
        p = stpcpy(p, "Content-Type: application/json\r\n");
        return send_response(obuf_.p, p, json.toString());
        
    } else if (msg_.method == kHttpPost) {
        // POST: Apply LoRA adapters by ID and scale (upstream llama.cpp format)
        
        // Validate content type
        if (!HasHeader(kHttpContentType) ||
            !IsMimeType(HeaderData(kHttpContentType),
                       HeaderLength(kHttpContentType),
                       "application/json")) {
            return send_error(400, "Content-Type must be application/json");
        }
        
        // Read the payload
        if (!read_payload())
            return false;
        
        // Parse JSON payload - expecting an array of {id, scale} objects
        auto [status, json] = Json::parse(std::string(payload_));
        if (status != Json::success)
            return send_error(400, Json::StatusToString(status));
        if (!json.isArray())
            return send_error(400, "Request body must be an array");
        
        // Apply the LoRA configuration
        return handle_upstream_lora_apply(json);
        
    } else {
        return send_error(405, "Method Not Allowed");
    }
}

bool
Client::handle_apply_adapters(Json& json)
{
    // Get active slots and apply current adapters to them
    if (::g_lora_adapters.empty()) {
        Json response;
        response["success"] = false;
        response["message"] = "No adapters loaded to apply";
        
        char* p = append_http_response_message(obuf_.p, 400);
        p = stpcpy(p, "Content-Type: application/json\r\n");
        return send_response(obuf_.p, p, response.toString());
    }
    
    // Apply adapters to all slots via the server using llama.cpp unified function
    SLOG("applying %d LoRA adapter(s) to all active slots using llama.cpp unified function", 
         (int)::g_lora_adapters.size());
    
    // Apply to all active slots
    Slots* slots = worker_->server_->slots_;
    pthread_mutex_lock(&slots->lock_);
    
    for (size_t i = 0; i < slots->slots_.size(); ++i) {
        Slot* slot = slots->slots_[i].get();
        if (slot->ctx_) {
            SLOG("applying LoRA adapters to slot #%d", slot->id_);
            llama_lora_adapters_apply(slot->ctx_, ::g_lora_adapters);
            
            // CRITICAL: Mark slot for refresh to handle LoRA changes properly
            // The slot's prefill() mechanism will intelligently preserve system prompts
            // and only re-evaluate what's necessary when the next request comes in
            slot->mark_for_refresh();
            SLOG("marked slot #%d for refresh after LoRA application", slot->id_);
        }
    }
    
    pthread_mutex_unlock(&slots->lock_);
    
    Json response;
    response["success"] = true;
    response["message"] = "Adapters applied to active slots";
    response["adapters_applied"] = (int)::g_lora_adapters.size();
    
    char* p = append_http_response_message(obuf_.p, 200);
    p = stpcpy(p, "Content-Type: application/json\r\n");
    return send_response(obuf_.p, p, response.toString());
}

bool
Client::handle_load_adapter(Json& json)
{
    // Load a new adapter from file
    if (!json.contains("path")) {
        return send_error(400, "Missing 'path' field for load operation");
    }
    
    std::string adapter_path = json["path"].getString();
    float scale = json.contains("scale") ? json["scale"].getNumber() : 1.0f;
    
    // Check if file exists
    if (!std::filesystem::exists(adapter_path)) {
        Json response;
        response["success"] = false;
        response["message"] = "Adapter file not found: " + adapter_path;
        
        char* p = append_http_response_message(obuf_.p, 404);
        p = stpcpy(p, "Content-Type: application/json\r\n");
        return send_response(obuf_.p, p, response.toString());
    }
    
    // Load the adapter
    char scale_buf[32];
    snprintf(scale_buf, sizeof(scale_buf), "%.2f", scale);
    SLOG("loading LoRA adapter from %s with scale %s", adapter_path.c_str(), scale_buf);
    
    struct llama_lora_adapter* adapter = llama_lora_adapter_init(model_, adapter_path.c_str());
    if (!adapter) {
        Json response;
        response["success"] = false;
        response["message"] = "Failed to load adapter from " + adapter_path;
        
        char* p = append_http_response_message(obuf_.p, 500);
        p = stpcpy(p, "Content-Type: application/json\r\n");
        return send_response(obuf_.p, p, response.toString());
    }
    
    // Create the adapter container
    llama_lora_adapter_container adapter_container;
    adapter_container.path = adapter_path;
    adapter_container.scale = scale;
    adapter_container.adapter = adapter;
    
    // Store the adapter
    int index = (int)::g_lora_adapters.size();
    ::g_lora_adapters.push_back(adapter_container);
    
    SLOG("successfully loaded LoRA adapter #%d from %s", index, adapter_path.c_str());
    
    Json response;
    response["success"] = true;
    response["message"] = "Adapter loaded successfully";
    response["index"] = index;
    response["path"] = adapter_path;
    response["scale"] = scale;
    response["total_adapters"] = (int)::g_lora_adapters.size();
    
    char* p = append_http_response_message(obuf_.p, 200);
    p = stpcpy(p, "Content-Type: application/json\r\n");
    return send_response(obuf_.p, p, response.toString());
}

bool
Client::handle_clear_adapters()
{
    // Clear all loaded adapters
    SLOG("clearing all %d LoRA adapter(s)", (int)::g_lora_adapters.size());
    
    int cleared_count = (int)::g_lora_adapters.size();
    for (auto& la : ::g_lora_adapters) {
        if (la.adapter) {
            llama_lora_adapter_free(la.adapter);
        }
    }
    
    ::g_lora_adapters.clear();
    
    SLOG("cleared %d LoRA adapter(s)", cleared_count);
    
    Json response;
    response["success"] = true;
    response["message"] = "All adapters cleared";
    response["cleared_count"] = cleared_count;
    response["remaining_count"] = 0;
    
    char* p = append_http_response_message(obuf_.p, 200);
    p = stpcpy(p, "Content-Type: application/json\r\n");
    return send_response(obuf_.p, p, response.toString());
}

bool
Client::handle_upstream_lora_apply(Json& json)
{
    // Handle upstream llama.cpp LoRA API format: array of {id, scale} objects
    std::vector<Json>& json_array = json.getArray();
    SLOG("applying LoRA configuration with %d entries", (int)json_array.size());
    
    // Process each entry in the array
    for (size_t i = 0; i < json_array.size(); i++) {
        Json& entry = json_array[i];
        
        if (!entry.isObject()) {
            return send_error(400, "Each entry must be an object with 'id' and 'scale' fields");
        }
        
        if (!entry.contains("id") || !entry.contains("scale")) {
            return send_error(400, "Each entry must have 'id' and 'scale' fields");
        }
        
        int id = entry["id"].getNumber();
        float scale = entry["scale"].getNumber();
        
        // Validate ID range
        if (id < 0 || id >= (int)::g_lora_adapters.size()) {
            return send_error(400, "Invalid adapter ID");
        }
        
        // Update the adapter configuration
        ::g_lora_adapters[id].scale = scale;
        
        char scale_buf[32];
        snprintf(scale_buf, sizeof(scale_buf), "%.2f", scale);
        SLOG("set LoRA adapter %d ('%s') scale to %s", 
             id, ::g_lora_adapters[id].path.c_str(), scale_buf);
    }
    
    // Re-apply LoRA adapters to all active slots with updated scales using llama.cpp unified function
    SLOG("re-applying LoRA adapters to all active slots using llama.cpp unified function");
    Slots* slots = worker_->server_->slots_;
    
    // Lock the slots to prevent concurrent access during LoRA re-application
    pthread_mutex_lock(&slots->lock_);
    
    for (size_t i = 0; i < slots->slots_.size(); ++i) {
        Slot* slot = slots->slots_[i].get();
        if (slot->ctx_) {
            SLOG("re-applying LoRA adapters to slot #%d", slot->id_);
            llama_lora_adapters_apply(slot->ctx_, ::g_lora_adapters);
            
            // CRITICAL: Mark slot for refresh to handle LoRA changes properly
            // The slot's prefill() mechanism will intelligently preserve system prompts
            // and only re-evaluate what's necessary when the next request comes in
            slot->mark_for_refresh();
            SLOG("marked slot #%d for refresh after LoRA update", slot->id_);
        }
    }
    
    pthread_mutex_unlock(&slots->lock_);
    SLOG("finished re-applying LoRA adapters to all slots");
    
    // Return updated adapter configuration
    Json response;
    response.setArray();
    std::vector<Json>& response_array = response.getArray();
    
    for (size_t i = 0; i < ::g_lora_adapters.size(); i++) {
        Json adapter;
        adapter.setObject();
        adapter["id"] = (int)i;
        adapter["path"] = ::g_lora_adapters[i].path;
        adapter["scale"] = ::g_lora_adapters[i].scale;
        response_array.push_back(adapter);
    }
    
    char* p = append_http_response_message(obuf_.p, 200);
    p = stpcpy(p, "Content-Type: application/json\r\n");
    return send_response(obuf_.p, p, response.toString());
}

} // namespace server
} // namespace lf
