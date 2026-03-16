We provide example llamafiles for a variety of models, so you can easily try out llamafile 
with different kinds of LLMs. The following table lists llamafiles bundled with the latest
available version of the server (v0.10.0). The smaller the file is, the more easily it will
run on your computer, even if no GPU is present (as a reference, Qwen3.5 0.8B Q8 generates
text on a Raspberry Pi5 at ~8 tokens/sec).

<table id="model-table">
  <thead>
    <tr>
      <th onclick="sortTable(0)" style="cursor:pointer">Model ⬍</th>
      <th onclick="sortTable(1)" style="cursor:pointer">Size ⬍</th>
      <th>License</th>
      <th>llamafile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td><a href="https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509">Apertus 8B Instruct 2509</a></td>
        <td data-val="5.9">5.9 GB</td>
        <td><a href="https://choosealicense.com/licenses/apache-2.0/">Apache 2.0</a></td>
        <td><a href="https://huggingface.co/mozilla-ai/llamafile_0.10.0/resolve/main/Apertus-8B-Instruct-2509.llamafile">Apertus-8B-Instruct-2509.llamafile</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/openai/gpt-oss-20b">gpt-oss 20b</a> mxfp4</td>
        <td data-val="12">12 GB</td>
        <td><a href="https://choosealicense.com/licenses/apache-2.0/">Apache 2.0</a></td>
        <td><a href="https://huggingface.co/mozilla-ai/llamafile_0.10.0/resolve/main/gpt-oss-20b-mxfp4.llamafile">gpt-oss-20b-mxfp4.llamafile</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/openai/gpt-oss-20b">gpt-oss 20b</a> Q5_K_S</td>
        <td data-val="12">12 GB</td>
        <td><a href="https://choosealicense.com/licenses/apache-2.0/">Apache 2.0</a></td>
        <td><a href="https://huggingface.co/mozilla-ai/llamafile_0.10.0/resolve/main/gpt-oss-20b-Q5_K_S.llamafile">gpt-oss-20b-Q5_K_S.llamafile</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/LiquidAI/LFM2-24B-A2B">LFM2 24B A2B</a> Q5_K_M</td>
        <td data-val="16">16 GB</td>
        <td><a href="https://huggingface.co/LiquidAI/LFM2-24B-A2B/blob/main/LICENSE">lfm1.0</a></td>
        <td><a href="https://huggingface.co/mozilla-ai/llamafile_0.10.0/resolve/main/LFM2-24B-A2B-Q5_K_M.llamafile">LFM2-24B-A2B-Q5_K_M.llamafile</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b">llava v1.6 mistral 7b</a> Q4_K_M</td>
        <td data-val="5.3">5.3 GB</td>
        <td><a href="https://choosealicense.com/licenses/apache-2.0/">Apache 2.0</a></td>
        <td><a href="https://huggingface.co/mozilla-ai/llamafile_0.10.0/resolve/main/llava-v1.6-mistral-7b-Q4_K_M.llamafile">llava-v1.6-mistral-7b-Q4_K_M.llamafile</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b">llava v1.6 mistral 7b</a> Q8_0</td>
        <td data-val="8.4">8.4 GB</td>
        <td><a href="https://choosealicense.com/licenses/apache-2.0/">Apache 2.0</a></td>
        <td><a href="https://huggingface.co/mozilla-ai/llamafile_0.10.0/resolve/main/llava-v1.6-mistral-7b-Q8_0.llamafile">llava-v1.6-mistral-7b-Q8_0.llamafile</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512">Ministral 3 3B Instruct 2512</a> Q4_K_M</td>
        <td data-val="3.4">3.4 GB</td>
        <td><a href="https://choosealicense.com/licenses/apache-2.0/">Apache 2.0</a></td>
        <td><a href="https://huggingface.co/mozilla-ai/llamafile_0.10.0/resolve/main/Ministral-3-3B-Instruct-2512-Q4_K_M.llamafile">Ministral-3-3B-Instruct-2512-Q4_K_M.llamafile</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512">Ministral 3 3B Instruct 2512</a> BF16</td>
        <td data-val="7.8">7.8 GB</td>
        <td><a href="https://choosealicense.com/licenses/apache-2.0/">Apache 2.0</a></td>
        <td><a href="https://huggingface.co/mozilla-ai/llamafile_0.10.0/resolve/main/Ministral-3-3B-Instruct-2512-BF16.llamafile">Ministral-3-3B-Instruct-2512-BF16.llamafile</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/Qwen/Qwen3.5-0.8B">Qwen3.5 0.8B</a> Q8_0</td>
        <td data-val="1.6">1.6 GB</td>
        <td><a href="https://choosealicense.com/licenses/apache-2.0/">Apache 2.0</a></td>
        <td><a href="https://huggingface.co/mozilla-ai/llamafile_0.10.0/resolve/main/Qwen3.5-0.8B-Q8_0.llamafile">Qwen3.5-0.8B-Q8_0.llamafile</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/Qwen/Qwen3.5-2B">Qwen3.5 2B</a> Q8_0</td>
        <td data-val="3.2">3.2 GB</td>
        <td><a href="https://choosealicense.com/licenses/apache-2.0/">Apache 2.0</a></td>
        <td><a href="https://huggingface.co/mozilla-ai/llamafile_0.10.0/resolve/main/Qwen3.5-2B-Q8_0.llamafile">Qwen3.5-2B-Q8_0.llamafile</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/Qwen/Qwen3.5-4B">Qwen3.5 4B</a> Q5_K_S</td>
        <td data-val="4.1">4.1 GB</td>
        <td><a href="https://choosealicense.com/licenses/apache-2.0/">Apache 2.0</a></td>
        <td><a href="https://huggingface.co/mozilla-ai/llamafile_0.10.0/resolve/main/Qwen3.5-4B-Q5_K_S.llamafile">Qwen3.5-4B-Q5_K_S.llamafile</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/Qwen/Qwen3.5-9B">Qwen3.5 9B</a> Q5_K_S</td>
        <td data-val="7.4">7.4 GB</td>
        <td><a href="https://choosealicense.com/licenses/apache-2.0/">Apache 2.0</a></td>
        <td><a href="https://huggingface.co/mozilla-ai/llamafile_0.10.0/resolve/main/Qwen3.5-9B-Q5_K_S.llamafile">Qwen3.5-9B-Q5_K_S.llamafile</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/Qwen/Qwen3.5-27B">Qwen3.5 27B</a> Q5_K_S</td>
        <td data-val="19">19 GB</td>
        <td><a href="https://choosealicense.com/licenses/apache-2.0/">Apache 2.0</a></td>
        <td><a href="https://huggingface.co/mozilla-ai/llamafile_0.10.0/resolve/main/Qwen3.5-27B-Q5_K_S.llamafile">Qwen3.5-27B-Q5_K_S.llamafile</a></td>
    </tr>
  </tbody>
</table>

<style>
#model-table th[onclick] { cursor: pointer; user-select: none; }
#model-table th[onclick]:hover { opacity: 0.8; }
</style>

<script>
(function () {
  let sortCol = -1, sortDir = 1;
  const labels = ['Model', 'Size', 'License', 'llamafile'];
  function sortTable(col) {
    const table = document.getElementById('model-table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const ths = table.querySelectorAll('th');
    if (sortCol === col) sortDir *= -1; else { sortCol = col; sortDir = 1; }
    ths.forEach((th, i) => {
      if (i === 0 || i === 1) {
        th.textContent = labels[i] + (i === col ? (sortDir === 1 ? ' ▲' : ' ▼') : ' ⬍');
      }
    });
    rows.sort((a, b) => {
      const ac = a.cells[col], bc = b.cells[col];
      const av = col === 1 ? parseFloat(ac.dataset.val) : ac.innerText.trim();
      const bv = col === 1 ? parseFloat(bc.dataset.val) : bc.innerText.trim();
      return (typeof av === 'number' ? av - bv : av.localeCompare(bv)) * sortDir;
    });
    rows.forEach(r => tbody.appendChild(r));
  }
  window.sortTable = sortTable;
  sortTable(1);
})();
</script>


## Legacy llamafiles

If you prefer the "classic llamafile experience" from previous versions (0.9.*),
here's a list of llamafiles bundled with the older server executable.


| Model                   | Size     | License                                                                                                                            | llamafile                                                                                                                                                                                      | other quants                                                                        |
| ---                     | ---      | ---                                                                                                                                | ---                                                                                                                                                                                            | ---                                                                                 |
| LLaMA 3.2 1B Instruct   | 1.11 GB  | [LLaMA 3.2](https://huggingface.co/Mozilla/Llama-3.2-1B-Instruct-llamafile/blob/main/LICENSE)                                      | [Llama-3.2-1B-Instruct-Q6\_K.llamafile](https://huggingface.co/Mozilla/Llama-3.2-1B-Instruct-llamafile/blob/main/Llama-3.2-1B-Instruct-Q6_K.llamafile?download=true)                           | [See HF repo](https://huggingface.co/Mozilla/Llama-3.2-1B-Instruct-llamafile)       |
| LLaMA 3.2 3B Instruct   | 2.62 GB  | [LLaMA 3.2](https://huggingface.co/Mozilla/Llama-3.2-3B-Instruct-llamafile/blob/main/LICENSE)                                      | [Llama-3.2-3B-Instruct.Q6\_K.llamafile](https://huggingface.co/Mozilla/Llama-3.2-3B-Instruct-llamafile/blob/main/Llama-3.2-3B-Instruct.Q6_K.llamafile?download=true)                           | [See HF repo](https://huggingface.co/Mozilla/Llama-3.2-3B-Instruct-llamafile)       |
| LLaMA 3.1 8B Instruct   | 5.23 GB  | [LLaMA 3.1](https://huggingface.co/Mozilla/Meta-Llama-3.1-8B-Instruct-llamafile/blob/main/LICENSE)                                 | [Llama-3.1-8B-Instruct.Q4\_K\_M.llamafile](https://huggingface.co/Mozilla/Meta-Llama-3.1-8B-Instruct-llamafile/resolve/main/Meta-Llama-3.1-8B-Instruct.Q4_K_M.llamafile?download=true)         | [See HF repo](https://huggingface.co/Mozilla/Meta-Llama-3.1-8B-Instruct-llamafile)  |
| Gemma 3 1B Instruct     | 1.32 GB  | [Gemma 3](https://ai.google.dev/gemma/terms)                                                                                       | [gemma-3-1b-it.Q6\_K.llamafile](https://huggingface.co/Mozilla/gemma-3-1b-it-llamafile/resolve/main/google_gemma-3-1b-it-Q6_K.llamafile?download=true)                                         | [See HF repo](https://huggingface.co/Mozilla/gemma-3-1b-it-llamafile)               |
| Gemma 3 4B Instruct     | 3.50 GB  | [Gemma 3](https://ai.google.dev/gemma/terms)                                                                                       | [gemma-3-4b-it.Q6\_K.llamafile](https://huggingface.co/Mozilla/gemma-3-4b-it-llamafile/resolve/main/google_gemma-3-4b-it-Q6_K.llamafile?download=true)                                         | [See HF repo](https://huggingface.co/Mozilla/gemma-3-4b-it-llamafile)               |
| Gemma 3 12B Instruct    | 7.61 GB  | [Gemma 3](https://ai.google.dev/gemma/terms)                                                                                       | [gemma-3-12b-it.Q4\_K\_M.llamafile](https://huggingface.co/Mozilla/gemma-3-12b-it-llamafile/resolve/main/google_gemma-3-12b-it-Q4_K_M.llamafile?download=true)                                 | [See HF repo](https://huggingface.co/Mozilla/gemma-3-12b-it-llamafile)              |
| QwQ 32B                 | 7.61 GB  | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                                                                      | [Qwen\_QwQ-32B-Q4\_K\_M.llamafile](https://huggingface.co/Mozilla/QwQ-32B-llamafile/resolve/main/Qwen_QwQ-32B-Q4_K_M.llamafile?download=true)                                                  | [See HF repo](https://huggingface.co/Mozilla/QwQ-32B-llamafile)                     |
| R1 Distill Qwen 14B     | 9.30 GB  | [MIT](https://choosealicense.com/licenses/mit/)                                                                                    | [DeepSeek-R1-Distill-Qwen-14B-Q4\_K\_M](https://huggingface.co/Mozilla/DeepSeek-R1-Distill-Qwen-14B-llamafile/resolve/main/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.llamafile?download=true)        | [See HF repo](https://huggingface.co/Mozilla/DeepSeek-R1-Distill-Qwen-14B-llamafile)|
| R1 Distill Llama 8B     | 5.23 GB  | [MIT](https://choosealicense.com/licenses/mit/)                                                                                    | [DeepSeek-R1-Distill-Llama-8B-Q4\_K\_M](https://huggingface.co/Mozilla/DeepSeek-R1-Distill-Llama-8B-llamafile/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.llamafile?download=true)        | [See HF repo](https://huggingface.co/Mozilla/DeepSeek-R1-Distill-Llama-8B-llamafile)|
| LLaVA 1.5               | 3.97 GB  | [LLaMA 2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)                                                     | [llava-v1.5-7b-q4.llamafile](https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile/resolve/main/llava-v1.5-7b-q4.llamafile?download=true)                                                     | [See HF repo](https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile)               |
| Mistral-7B-Instruct v0.3| 4.42 GB  | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                                                                      | [mistral-7b-instruct-v0.3.Q4\_0.llamafile](https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.3-llamafile/resolve/main/Mistral-7B-Instruct-v0.3.Q4_0.llamafile?download=true)               | [See HF repo](https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.3-llamafile)    |
| Granite 3.2 8B Instruct | 5.25 GB  | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                                                                      | [granite-3.2-8b-instruct-Q4\_K\_M.llamafile](https://huggingface.co/Mozilla/granite-3.2-8b-instruct-llamafile/resolve/main/granite-3.2-8b-instruct-Q4_K_M.llamafile?download=true)             | [See HF repo](https://huggingface.co/Mozilla/granite-3.2-8b-instruct-llamafile)     |
| Phi-3-mini-4k-instruct  | 7.67 GB  | [Apache 2.0](https://huggingface.co/Mozilla/Phi-3-mini-4k-instruct-llamafile/blob/main/LICENSE)                                    | [Phi-3-mini-4k-instruct.F16.llamafile](https://huggingface.co/Mozilla/Phi-3-mini-4k-instruct-llamafile/resolve/main/Phi-3-mini-4k-instruct.F16.llamafile?download=true)                        | [See HF repo](https://huggingface.co/Mozilla/Phi-3-mini-4k-instruct-llamafile)      |
| Mixtral-8x7B-Instruct   | 30.03 GB | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                                                                      | [mixtral-8x7b-instruct-v0.1.Q5\_K\_M.llamafile](https://huggingface.co/Mozilla/Mixtral-8x7B-Instruct-v0.1-llamafile/resolve/main/mixtral-8x7b-instruct-v0.1.Q5_K_M.llamafile?download=true)    | [See HF repo](https://huggingface.co/Mozilla/Mixtral-8x7B-Instruct-v0.1-llamafile)  |
| OLMo-7B                 | 5.68 GB  | [Apache 2.0](https://huggingface.co/Mozilla/OLMo-7B-0424-llamafile/blob/main/LICENSE)                                              | [OLMo-7B-0424.Q6\_K.llamafile](https://huggingface.co/Mozilla/OLMo-7B-0424-llamafile/resolve/main/OLMo-7B-0424.Q6_K.llamafile?download=true)                                                   | [See HF repo](https://huggingface.co/Mozilla/OLMo-7B-0424-llamafile)                |
| *Text Embedding Models* |          |                                                                                                                                    |                                                                                                                                                                                                |                                                                                     |
| E5-Mistral-7B-Instruct  | 5.16 GB  | [MIT](https://choosealicense.com/licenses/mit/)                                                                                    | [e5-mistral-7b-instruct-Q5_K_M.llamafile](https://huggingface.co/Mozilla/e5-mistral-7b-instruct/resolve/main/e5-mistral-7b-instruct-Q5_K_M.llamafile?download=true)                            | [See HF repo](https://huggingface.co/Mozilla/e5-mistral-7b-instruct)                |
| mxbai-embed-large-v1    | 0.7 GB   | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                                                                      | [mxbai-embed-large-v1-f16.llamafile](https://huggingface.co/Mozilla/mxbai-embed-large-v1-llamafile/resolve/main/mxbai-embed-large-v1-f16.llamafile?download=true)                              | [See HF Repo](https://huggingface.co/Mozilla/mxbai-embed-large-v1-llamafile)        |


## Running llamafile in CLI mode

Here is an example for the Mistral command-line llamafile:

```sh
./mistral-7b-instruct-v0.2.Q5_K_M.llamafile --temp 0.7 -p '[INST]Write a story about llamas[/INST]'
```

And here is an example for WizardCoder-Python command-line llamafile:

```sh
./wizardcoder-python-13b.llamafile --temp 0 -e -r '```\n' -p '```c\nvoid *memcpy_sse2(char *dst, const char *src, size_t size) {\n'
```

And here's an example for the LLaVA command-line llamafile:

```sh
./llava-v1.5-7b-q4.llamafile --temp 0.2 --image lemurs.jpg -e -p '### User: What do you see?\n### Assistant:'
```

As described in the [Getting Started](quickstart.md) section, 
macOS, Linux, and BSD users will need to use the "chmod"
command to grant execution permissions to the file before running these
llamafiles for the first time.

Unfortunately, Windows users cannot make use of many of these example
llamafiles because Windows has a maximum executable file size of 4GB,
and all of these examples exceed that size. (The LLaVA llamafile works
on Windows because it is 30MB shy of the size limit.) But don't lose
heart: llamafile allows you to use external weights; this is described
in the [Getting Started](quickstart.md) section.

**Having trouble? See the [Troubleshooting](troubleshooting.md) page.**


## A note about models

The example llamafiles provided above should not be interpreted as
endorsements or recommendations of specific models, licenses, or data
sets on the part of Mozilla.
