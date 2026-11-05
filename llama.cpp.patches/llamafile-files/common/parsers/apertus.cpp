#include "parsers.h"

// Apertus 1.5 format:
// - Reasoning: <|inner_prefix|>{reasoning}<|inner_suffix|>, tool calls can come before <|inner_suffix|>
// - Tool calls: <|tools_prefix|>[{"name": {arguments}}, ...]<|tools_suffix|>
// - Tool outputs: <|tool_output_start|>{output}<|tool_output_end|>

static const std::string APERTUS_ASSISTANT_START = "<|assistant_start|>";
static const std::string APERTUS_INNER_START     = "<|inner_prefix|>";
static const std::string APERTUS_INNER_END       = "<|inner_suffix|>";
static const std::string APERTUS_TOOLS_START     = "<|tools_prefix|>";
static const std::string APERTUS_TOOLS_END       = "<|tools_suffix|>";

static std::string apertus_text(const json & content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }
    std::string text;
    if (content.is_array()) {
        for (const auto & part : content) {
            if (part.is_object() && part.value("type", "") == "text" && part.contains("text")) {
                text += part.at("text").get<std::string>();
            }
        }
    }
    return text;
}

// the template renders past reasoning only from content.blocks and rejects a mix of string and
// block assistant messages, so all assistant messages use blocks when one of them has reasoning
static json apertus_adjust_messages(const json & messages) {
    bool has_reasoning = false;
    for (const auto & msg : messages) {
        if (msg.value("role", "") == "assistant" && !apertus_text(msg.value("reasoning_content", json())).empty()) {
            has_reasoning = true;
            break;
        }
    }
    if (!has_reasoning) {
        return messages;
    }

    json adjusted = json::array();
    for (const auto & msg : messages) {
        if (msg.value("role", "") != "assistant") {
            adjusted.push_back(msg);
            continue;
        }

        json blocks = json::array();
        const auto reasoning = apertus_text(msg.value("reasoning_content", json()));
        if (!reasoning.empty()) {
            blocks.push_back({ { "type", "thoughts" }, { "text", reasoning } });
        }
        const auto content = apertus_text(msg.value("content", json()));
        if (!content.empty()) {
            blocks.push_back({ { "type", "response" }, { "text", content } });
        }
        if (msg.contains("tool_calls") && msg.at("tool_calls").is_array() && !msg.at("tool_calls").empty()) {
            blocks.push_back({ { "type", "tool_calls" }, { "calls", msg.at("tool_calls") } });
        }

        json converted = msg;
        converted["content"] = { { "blocks", blocks } };
        converted.erase("reasoning_content");
        converted.erase("tool_calls");
        adjusted.push_back(converted);
    }
    return adjusted;
}

common_chat_params common_chat_params_init_apertus(const common_chat_template &          tmpl,
                                                   const autoparser::generation_params & inputs) {
    common_chat_params data;

    const auto messages = apertus_adjust_messages(inputs.messages);

    data.prompt             = common_chat_template_direct_apply_impl(tmpl, inputs, messages);
    data.generation_prompt  = common_chat_template_generation_prompt_impl(tmpl, inputs, messages);
    data.format             = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking  = true;
    data.thinking_start_tag = APERTUS_INNER_START;
    data.thinking_end_tags  = { APERTUS_INNER_END };
    data.preserved_tokens   = {
        APERTUS_INNER_START,
        APERTUS_INNER_END,
        APERTUS_TOOLS_START,
        APERTUS_TOOLS_END,
    };

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, APERTUS_ASSISTANT_START },
        { COMMON_CHAT_ROLE_TOOL,      "<|tool_output_start|>" },
        { COMMON_CHAT_ROLE_USER,      "<|user_start|>"        },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|system_start|>"      },
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto use_tools           = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    auto include_grammar     = has_response_format || use_tools;

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = APERTUS_ASSISTANT_START;
        if (!msg.reasoning_content.empty()) {
            data.generation_prompt += APERTUS_INNER_START + msg.reasoning_content;
            if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
                data.generation_prompt += APERTUS_INNER_END;
            }
        }
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += msg.render_content();
        }

        data.prompt.append(data.generation_prompt);
    }

    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(APERTUS_ASSISTANT_START);

        auto reasoning = p.eps();
        if (extract_reasoning) {
            // a tool call can end the reasoning without <|inner_suffix|>
            reasoning = p.optional(p.literal(APERTUS_INNER_START) +
                                   p.reasoning(p.until_one_of({ APERTUS_INNER_END, APERTUS_TOOLS_START })) +
                                   p.optional(p.literal(APERTUS_INNER_END)));
        }

        if (has_response_format) {
            return generation_prompt + reasoning + p.content(p.schema(p.json(), "response-format", inputs.json_schema)) + p.end();
        }

        if (use_tools) {
            // <|tools_suffix|> is an EOG token, keep it out of the tool call grammar:
            // the grammar only allows EOG tokens once it is complete
            auto tool_calls = p.standard_json_tools(
                APERTUS_TOOLS_START, /* section_end = */ "", inputs.tools, inputs.parallel_tool_calls,
                inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED,
                /* name_key = */ "", /* args_key = */ "",
                /* array_wrapped = */ true, /* function_is_key = */ true);

            return generation_prompt + reasoning + p.optional(p.content(p.until(APERTUS_TOOLS_START))) + tool_calls +
                   p.optional(p.literal(APERTUS_TOOLS_END)) + p.end();
        }

        return generation_prompt + reasoning + p.content(p.rest()) + p.end();
    });

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || (use_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            data.parser.build_grammar(builder, data.grammar_lazy);
        });

        if (use_tools) {
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, APERTUS_TOOLS_START },
            };
        }
    }

    return data;
}
