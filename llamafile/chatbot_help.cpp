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

#include "chatbot.h"

#include <vector>

#include "color.h"

namespace lf {
namespace chatbot {

void on_help(const std::vector<std::string> &args) {
    if (args.size() == 1) {
        fprintf(stderr, "\
" BOLD "available commands" RESET "\n\
  ctrl-j                   insert line in multi-line mode\n\
  \"\"\"                      use triple quotes for multi-line input\n\
  /clear                   restart conversation\n\
  /context                 print context window usage\n\
  /dump [FILE]             print or save context window to file\n\
  /exit                    end program\n\
  /forget                  erase oldest message from context\n\
  /help [COMMAND]          show help\n\
  /manual [on|off]         toggle manual role mode\n\
  /pop                     restore context window size\n\
  /push                    push context window size to stack\n\
  /stack                   prints context window stack\n\
  /stats                   print performance metrics\n\
  /undo                    erases last message in conversation\n\
  /upload FILE             share image or text file with assistant\n\
");
    } else if (args[1] == "context") {
        fprintf(stderr, "\
usage: /context" RESET "\n\
prints information about context window usage. this helps you know how\n\
soon you're going to run out of tokens for the current conversation.\n\
");
    } else if (args[1] == "dump") {
        fprintf(stderr, "\
" BOLD "usage: /dump [FILE]" RESET "\n\
dumps raw tokens for current conversation history. special tokens are\n\
printed in the a model specific chat syntax. this is useful for seeing\n\
specifically what data is being evaluated by the model. by default it\n\
will be printed to the terminal. if a FILE argument is specified, then\n\
the raw conversation history will be written to that filename.\n\
");
    } else if (args[1] == "exit") {
        fprintf(stderr, "\
" BOLD "usage: /exit" RESET "\n\
this command will cause the process to exit. it is essentially the same\n\
as typing ctrl-d which signals an eof condition. it also does the same\n\
thing as typing ctrl-c when the >>> user input prompt is displayed.\n\
");
    } else if (args[1] == "manual") {
        fprintf(stderr, "\
" BOLD "usage: /manual [on|off]" RESET "\n\
puts the chatbot in manual mode. this is useful if you want to inject\n\
a response as the model rather than the user. it's also possible to add\n\
additional system prompts to the conversation history. when the manual\n\
mode is activated, a hint is displayed next to the '>>>' indicating\n\
the current role, which can be 'user', 'assistant', or 'system'. if\n\
enter is pressed on an empty line, then llamafile will cycle between\n\
all three roles. when /manual is specified without an argument, it will\n\
toggle manual mode. otherwise an 'on' or 'off' argument is supplied.\n\
");
    } else if (args[1] == "help") {
        fprintf(stderr, "\
" BOLD "usage: /help [COMMAND]" RESET "\n\
shows help on how to issue commands to your llamafile. if no argument is\n\
specified, then a synopsis of all available commands will be printed. if\n\
a specific command name is given (e.g. /help dump) then documentation on\n\
the usage of that specific command will be printed.\n\
");
    } else if (args[1] == "stats") {
        fprintf(stderr, "\
" BOLD "usage: /stats" RESET "\n\
prints performance statistics for current session. this includes prompt\n\
evaluation time in tokens per second, which indicates prefill speed, or\n\
how quickly llamafile is able to read text. the 'eval time' statistic\n\
gives you prediction or token generation speed, in tokens per second,\n\
which tells you how quickly llamafile is able to write text.\n\
");
    } else if (args[1] == "clear") {
        fprintf(stderr, "\
usage: /clear" RESET "\n\
start conversation over from the beginning. this command adjusts the\n\
context window to what it was after the initial system prompt. this\n\
command also erases the /push stack.\n\
");
    } else if (args[1] == "push") {
        fprintf(stderr, "\
usage: /push" RESET "\n\
save current size of context window to stack. this command may be used\n\
with /pop to backtrack a conversation.\n\
");
    } else if (args[1] == "pop") {
        fprintf(stderr, "\
usage: /pop" RESET "\n\
restores size of context window from stack. this command may be used\n\
with /push to backtrack a conversation.\n\
");
    } else if (args[1] == "stack") {
        fprintf(stderr, "\
usage: /stack" RESET "\n\
prints the current conversation stack, created by /push commands.\n\
the stack consists of token offsets within the context window.\n\
");
    } else if (args[1] == "undo") {
        fprintf(stderr, "\
usage: /undo" RESET "\n\
erases last exchange in conversation. in the normal mode, this includes\n\
what the assistant last said, as well as the question that was asked. in\n\
manual mode, this will erase only the last chat message.\n\
");
    } else if (args[1] == "upload") {
        fprintf(stderr, "\
usage: /upload FILE" RESET "\n\
shares file from local hard drive with assistant. if this is a text file\n\
then a markdown system prompt is generated and added to the conversation\n\
history that gives the assistant readonly access to the file content and\n\
metadata. files with nul characters in them are currently not supported.\n\
image files (jpg/png/gif) may be uploaded if you specified a clip vision\n\
model (e.g. LLaVA) earlier when running llamafile with the --mmproj flag\n\
");
    } else if (args[1] == "forget") {
        fprintf(stderr, "\
usage: /forget" RESET "\n\
erase oldest chat message from context window. if you run out of context\n\
window, then this command can help you free up space. the oldest message\n\
excludes the original system prompt, with is preserved. this command may\n\
be run multiple times to erase multiple messages. there's also the /undo\n\
command which deletes the most recent chat message instead.\n\
");
    } else {
        fprintf(stderr, BRIGHT_RED "%s: unknown command" RESET "\n", args[1].c_str());
    }
}

} // namespace chatbot
} // namespace lf
