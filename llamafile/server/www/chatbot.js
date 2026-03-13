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

const API_KEY = "your-api-key-here";

const DEFAULT_SYSTEM_PROMPT =
      "A chat between a curious human and an artificial " +
      "intelligence assistant. The assistant gives helpful, " +
      "detailed, and polite answers to the human's questions.";

const DEFAULT_FLAGZ = {
  "model": null,
  "prompt": null,
  "nologo": false,
  "no_display_prompt": false,
  "frequency_penalty": 0,
  "presence_penalty": 0,
  "temperature": 0.8,
  "top_p": 0.95,
  "seed": null,
  "is_base_model": true,
  "completion_mode": false,
};

const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const stopButton = document.getElementById("stop-button");
const settingsButton = document.getElementById("settings-button");
const settingsModal = document.getElementById("settings-modal");
const closeSettings = document.getElementById("close-settings");
const redoButton = document.getElementById('redo-button');
const chatInterface = document.getElementById("chat-interface");
const completionsInterface = document.getElementById("completions-interface");
const completionsInput = document.getElementById("completions-input");
const completeButton = document.getElementById("complete-button");
const completionsSettingsButton = document.getElementById("completions-settings-button");
const completionsStopButton = document.getElementById("completions-stop-button");
const uploadButton = document.getElementById("upload-button");
const fileUpload = document.getElementById("file-upload");

let abortController = null;
let disableAutoScroll = false;
let streamingMessageContent = [];
let originalLength = 0;
let uploadedFiles = [];
let chatHistory = [];
let flagz = null;

function getUrlPrefix() {
    const path = window.location.pathname;
    return path.substring(0, path.lastIndexOf('/'));
}

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).substring(2);
}

function wrapMessageElement(messageElement, role) {
  const wrapper = document.createElement("div");
  wrapper.appendChild(messageElement);
  if (role == "assistant") {
    const controlContainer = wrapper.appendChild(document.createElement("div"));
    controlContainer.appendChild(createCopyButton(() => messageElement.textContent, () => messageElement.innerHTML));
    controlContainer.appendChild(infoButton(wrapper));
    controlContainer.classList.add("message-controls");
  }
  wrapper.classList.add("message-wrapper", role);
  return wrapper;
}

function infoButton(container, stats) {
  let button = container?.querySelector("#stats");
  let statsElement = container?.querySelector("#info-container");
  if (!button) {
    button = document.createElement("button");
    button.id = "stats";
    button.innerText = "i";
    button.style.fontFamily = "monospace";

    statsElement = document.createElement("div");
    statsElement.id = "info-container";
    statsElement.className = "hidden";
    container.append(statsElement);
    button.addEventListener("click", () => {
      const show = !button.classList.contains("toggled");
      statsElement.classList.toggle("hidden", !show);
      button.classList.toggle("toggled", show);
      if (show)
        requestAnimationFrame(() => scrollIntoViewIfNeeded(statsElement, container.parentElement));
    });
  }
  button.style.display = stats ? "" : "none";
  if (stats) {
    const parts = [];
    const promptDurationMs = stats.firstContentTime - stats.startTime;
    const responseDurationMs = stats.endTime - stats.firstContentTime;
    if (promptDurationMs > 0 && stats.promptTokenCount > 0) {
      const tokensPerSecond = (stats.promptTokenCount / (promptDurationMs / 1000)).toFixed(2);
      const durationString = promptDurationMs >= 1000 ? `${(promptDurationMs / 1000).toFixed(2)}s` : `${promptDurationMs}ms`;
      parts.push(`Processed ${stats.promptTokenCount} input tokens in ${durationString} (${tokensPerSecond} tokens/s)`);
    }
    if (responseDurationMs > 0 && stats.reponseTokenCount > 0) {
      const tokensPerSecond = (stats.reponseTokenCount / (responseDurationMs / 1000)).toFixed(2);
      const durationString = responseDurationMs >= 1000 ? `${(responseDurationMs / 1000).toFixed(2)}s` : `${promptDurationMs}ms`;
      parts.push(`Generated ${stats.reponseTokenCount} tokens in ${durationString} (${tokensPerSecond} tokens/s)`)
    } else {
      parts.push("Incomplete");
    }
    button.title = parts.join("\n");
    statsElement.innerHTML = "";
    parts.forEach(part => statsElement.appendChild(wrapInSpan(part + " ")));
  }
  return button;
}

function scrollIntoViewIfNeeded(elem, container) {
  let rectElem = elem.getBoundingClientRect(), rectContainer = container.getBoundingClientRect();
  if (rectElem.bottom > rectContainer.bottom) elem.scrollIntoView(false);
  if (rectElem.top < rectContainer.top) elem.scrollIntoView();
}

function wrapInSpan(innerText) {
  const span = document.createElement("span");
  span.innerText = innerText;
  return span;
}

function createMessageElement(content) {
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message");
  let hdom = new HighlightDom(messageDiv);
  const high = new RenderMarkdown(hdom);
  high.feed(content);
  high.flush();
  return messageDiv;
}

function scrollToBottom() {
  if (!disableAutoScroll) {
    document.getElementById("bottom").scrollIntoView({ behavior: "instant" });
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
}

function onChatInput() {
  chatInput.style.height = "auto";  // computes scrollHeight
  chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + "px";
}

function cleanupAfterMessage() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
  chatInput.disabled = false;
  sendButton.style.display = "inline-block";
  stopButton.style.display = "none";
  abortController = null;
  if (!disableAutoScroll) {
    scrollToBottom();
    chatInput.focus();
  }
  disableAutoScroll = false;
}

function onWheel(e) {
  if (e.deltaY == undefined || e.deltaY < 0)
    disableAutoScroll = true;
}

async function handleChatStream(response, stats) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let currentMessageElement = null;
  let currentMessageWrapper = null;
  let messageAppended = false;
  let finishReason = null;
  let hdom = null;
  let high = null;
  streamingMessageContent = [];
  const prefillStatus = document.getElementById('prefill-status');
  const progressBar = prefillStatus.querySelector('.progress-bar');

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");

      // process all complete lines
      for (let i = 0; i < lines.length - 1; i++) {
        const line = lines[i].trim();
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") {
            prefillStatus.style.display = "none";
            continue;
          }
          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices[0]?.delta?.content || "";
            finishReason = parsed.choices[0]?.finish_reason;

            // handle prefill progress
            if (parsed.x_prefill_progress !== undefined) {
              prefillStatus.style.display = "flex";
              progressBar.style.width = `${parsed.x_prefill_progress * 100}%`;
            } else {
              if (content && !stats.firstContentTime) {
                // Finished parsing the prompt
                stats.firstContentTime = Date.now();
                prefillStatus.style.display = "none";
              }
            }

            if (content && !messageAppended) {
              currentMessageElement = createMessageElement("");
              currentMessageWrapper = wrapMessageElement(currentMessageElement, "assistant");
              chatMessages.appendChild(currentMessageWrapper);
              hdom = new HighlightDom(currentMessageElement);
              high = new RenderMarkdown(hdom);
              messageAppended = true;
            }

            if (messageAppended && content) {
              streamingMessageContent.push(content);
              high.feed(content);
              scrollToBottom();
            }
            if (parsed.usage) {
              stats.endTime = Date.now()
              stats.promptTokenCount = parsed.usage.prompt_tokens
              stats.reponseTokenCount = parsed.usage.completion_tokens
            }
          } catch (e) {
            console.error("Error parsing JSON:", e);
          }
        }
      }

      // keep the last incomplete line in the buffer
      buffer = lines[lines.length - 1];
    }
  } catch (error) {
    if (error.name !== "AbortError") {
      console.error("Error reading stream:", error);
    }
  } finally {
    if (messageAppended) {
      stats.firstContentTime = stats.firstContentTime ?? Date.now();
      stats.endTime = stats.endTime ?? Date.now();
      infoButton(currentMessageWrapper, stats);
      high.flush();
      // we don't supply max_tokens, so "length" can
      // only mean that we ran out of context window
      if (finishReason === "length") {
        let img = document.createElement("IMG");
        img.className = "ooc";
        img.src = "ooc.svg";
        img.alt = "🚫";
        img.title = "Message truncated due to running out of context window. Consider tuning --ctx-size and/or --reserve-tokens";
        img.width = 16;
        img.height = 16;
        hdom.lastElement.appendChild(img);
      }
    }
    prefillStatus.style.display = "none";
    cleanupAfterMessage();
  }
}

function stopMessage() {
  if (abortController) {
    abortController.abort();
    cleanupAfterMessage();
  }
}

function fixUploads(str) {
  str = uploadedFiles.reduce(
    (text, [from, to]) => text.replaceAll(from, to),
    str);
  return str;
}

async function sendMessage() {
  const message = fixUploads(chatInput.value.trim());
  if (!message)
    return;

  // disable input while processing
  chatInput.value = "";
  chatInput.disabled = true;
  onChatInput();
  disableAutoScroll = false;
  sendButton.style.display = "none";
  stopButton.style.display = "inline-block";
  stopButton.focus();
  abortController = new AbortController();

  // add user message to chat
  const userMessageElement = createMessageElement(message);
  chatMessages.appendChild(wrapMessageElement(userMessageElement, "user"));
  scrollToBottom();

  // update chat history
  chatHistory.push({ role: "user", content: message });

  const settings = loadSettings();
  try {
    const stats = {
      startTime: Date.now(),      // Timestamp when the request started
      firstContentTime: null, // Timestamp when the first content was received
      endTime: null,        // Timestamp when the response was fully received
      promptTokenCount: 0,  // Number of tokens in the prompt
      reponseTokenCount: 0   // Number of tokens in the response
    };
    const response = await fetch(`${getUrlPrefix()}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${API_KEY}`
      },
      body: JSON.stringify({
        model: flagz.model || "gpt-3.5-turbo",
        messages: chatHistory,
        temperature: settings.temperature,
        top_p: settings.top_p,
        presence_penalty: settings.presence_penalty,
        frequency_penalty: settings.frequency_penalty,
        stream: true,
        stream_options: {
          include_usage: true
        }
      }),
      signal: abortController.signal
    });
    if (response.ok) {
      await handleChatStream(response, stats);
      const lastMessage = streamingMessageContent.join("");
      if (lastMessage)
        chatHistory.push({ role: "assistant", content: lastMessage });
    } else {
      console.error("sendMessage() failed due to server error", response);
      chatMessages.appendChild(wrapMessageElement(createMessageElement(
        `Server replied with error code ${response.status} ${response.statusText}`),
        "system"));
      cleanupAfterMessage();
    }
  } catch (error) {
    if (error.name !== "AbortError") {
      console.error("sendMessage() failed due to unexpected exception", error);
      chatMessages.appendChild(wrapMessageElement(createMessageElement(
        "There was an error processing your request."),
        "system"));
    }
    cleanupAfterMessage();
  }
}

function onDragBegin(e) {
  e.preventDefault();
  e.stopPropagation();
  chatInput.classList.add("drag-over");
}

function onDragEnd(e) {
  e.preventDefault();
  e.stopPropagation();
  chatInput.classList.remove("drag-over");
}

function onDrop(e) {
  const files = e.dataTransfer.files;
  [...files].forEach(onFile);
}

function onPaste(e) {
  const items = e.clipboardData.items;
  for (let item of items) {
    if (item.type.startsWith("image/")) {
      e.preventDefault();
      onFile(item.getAsFile());
      return;
    }
  }
}

// fixes image data uri
// - convert to jpg if it's not jpg/png/gif
// - reduce quality and/or downscale if too big
async function fixImageDataUri(dataUri, maxLength = 1024 * 1024) {
  const mimeMatch = dataUri.match(/^data:([^;,]+)/);
  if (!mimeMatch)
    throw new Error("bad image data uri");
  const mimeType = mimeMatch[1].toLowerCase();
  const supported = ["image/jpeg", "image/jpg", "image/png", "image/gif"];
  if (supported.includes(mimeType))
    if (dataUri.length <= maxLength)
      return dataUri;
  const lossless = ["image/png", "image/gif"];
  const quality = lossless.includes(mimeType) ? 0.92 : 0.8;
  function createScaledCanvas(img, scale) {
    const canvas = document.createElement("canvas");
    canvas.width = Math.floor(img.width * scale);
    canvas.height = Math.floor(img.height * scale);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.fillStyle = "white";  // in case of transparency
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas;
  }
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = async () => {
      let scale = 1.0;
      let attempts = 0;
      const maxAttempts = 5;
      const initialCanvas = createScaledCanvas(img, scale);
      let result = initialCanvas.toDataURL("image/jpeg", quality);
      while (result.length > maxLength && attempts < maxAttempts) {
        attempts++;
        scale *= 0.7071;
        const scaledCanvas = createScaledCanvas(img, scale);
        result = scaledCanvas.toDataURL("image/jpeg", quality);
        result.length = result.length;
      }
      if (result.length <= maxLength) {
        resolve(result);
      } else {
        reject(new Error(`Could not reduce image to ${(maxLength/1024).toFixed(2)}kb after ${maxAttempts} attempts`));
      }
    };
    img.onerror = () => {
      reject(new Error('Failed to load image from data URI'));
    };
    img.src = dataUri;
  });
}

async function onFile(file) {
  const reader = new FileReader();
  if (file.type.toLowerCase().startsWith('image/')) {
    reader.onloadend = async function() {
      const description = file.name;
      const realDataUri = await fixImageDataUri(reader.result);
      const fakeDataUri = 'data:,placeholder/' + generateId();
      uploadedFiles.push([fakeDataUri, realDataUri]);
      insertText(chatInput, `![${description}](${fakeDataUri})`);
    };
    reader.readAsDataURL(file);
  } else if (file.type.toLowerCase().startsWith('text/')) {
    reader.onloadend = function() {
      const content = reader.result;
      insertText(chatInput, `\`\`\`\n${content}\n\`\`\``);
    };
    reader.readAsText(file);
  } else {
    alert('Only image and text files are supported');
    return;
  }
}

function checkSurroundingNewlines(text, pos) {
  const beforeCaret = text.slice(0, pos);
  const afterCaret = text.slice(pos);
  const precedingNewlines = beforeCaret.match(/\n*$/)[0].length;
  const followingNewlines = afterCaret.match(/^\n*/)[0].length;
  return { precedingNewlines, followingNewlines };
}

function insertText(elem, text) {
  const pos = elem.selectionStart;
  const isCodeBlock = text.includes('```');

  if (isCodeBlock) {
    const { precedingNewlines, followingNewlines } = checkSurroundingNewlines(elem.value, pos);
    const needsLeadingNewlines = pos > 0 && precedingNewlines < 2 ? '\n'.repeat(2 - precedingNewlines) : '';
    const needsTrailingNewlines = pos < elem.value.length && followingNewlines < 2 ? '\n'.repeat(2 - followingNewlines) : '';
    text = needsLeadingNewlines + text + needsTrailingNewlines;
  }

  elem.value = elem.value.slice(0, pos) + text + elem.value.slice(pos);
  const newPos = pos + text.length;
  elem.setSelectionRange(newPos, newPos);
  elem.focus();
  elem.dispatchEvent(new Event("input"));
}

function onKeyDown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

async function fetchFlagz() {
  try {
    const response = await fetch(`${getUrlPrefix()}/flagz`);
    return await response.json();
  } catch (error) {
    console.error("Could not fetch /flagz so using defaults", error);
    return DEFAULT_FLAGZ;
  }
}

function getSystemPrompt() {
  let defaultPrompt = flagz.prompt;
  if (!defaultPrompt)
    defaultPrompt = DEFAULT_SYSTEM_PROMPT;
  let promptsText = localStorage.getItem("v1.prompts");
  if (!promptsText)
    return defaultPrompt;
  let prompts = JSON.parse(promptsText);
  let prompt = prompts[defaultPrompt];
  if (!prompt)
    return defaultPrompt;
  return prompt;
}

function updateModelInfo() {
  if (flagz.model) {
    const modelName = flagz.model;
    document.title = `${modelName} - llamafile`;
    document.getElementById("model").textContent = modelName;
    document.getElementById("model-completions").textContent = modelName;
  }
  if (!flagz.nologo) {
    document.querySelectorAll(".logo").forEach(logo => logo.style.display = "inline-block");
  }
}

function startChat(history) {
  chatHistory = history;
  chatMessages.innerHTML = "";
  for (let i = 0; i < chatHistory.length; i++) {
    if (flagz.no_display_prompt && chatHistory[i].role == "system")
      continue;
    chatMessages.appendChild(wrapMessageElement(createMessageElement(chatHistory[i].content),
      chatHistory[i].role));
  }
  scrollToBottom();
}

function loadSettings() {
  const stored = localStorage.getItem('v1.modelSettings');
  if (stored) {
    return JSON.parse(stored);
  }
  return {
    temperature: flagz.temperature,
    top_p: flagz.top_p,
    presence_penalty: flagz.presence_penalty,
    frequency_penalty: flagz.frequency_penalty,
  };
}

function saveSettings(settings) {
  localStorage.setItem('v1.modelSettings', JSON.stringify(settings));
}

function formatDoubleWithPlus(x) {
  return (x >= 0 ? "+" : "") + x.toFixed(2);
}

function updateSettingsDisplay(settings) {
  document.getElementById("temp-value").textContent = settings.temperature ? settings.temperature.toFixed(2) : "0.00 (deterministic)";
  document.getElementById("top-p-value").textContent = settings.top_p.toFixed(2);
  document.getElementById("presence-value").textContent = formatDoubleWithPlus(settings.presence_penalty);
  document.getElementById("frequency-value").textContent = formatDoubleWithPlus(settings.frequency_penalty);
  document.getElementById("temperature").value = settings.temperature;
  document.getElementById("top-p").value = settings.top_p;
  document.getElementById("presence-penalty").value = settings.presence_penalty;
  document.getElementById("frequency-penalty").value = settings.frequency_penalty;

  // Handle top-p disabling - using a more reliable selector
  const topPSettingItem = document.querySelector('.setting-item:has(#top-p)');
  if (settings.temperature === 0) {
    topPSettingItem.classList.add('disabled');
  } else {
    topPSettingItem.classList.remove('disabled');
  }

  // Update top-p description with percentage
  const topPDescription = topPSettingItem.querySelector('.setting-description');
  if (settings.top_p >= 1) {
    topPDescription.textContent = "Disabled. All tokens will be considered by the sampler.";
  } else if (settings.top_p > .5) {
    const percentage = Math.round((1 - settings.top_p) * 100);
    topPDescription.textContent = `The bottom ${percentage}% tokens will be ignored by the sampler.`;
  } else {
    const percentage = Math.round(settings.top_p * 100);
    topPDescription.textContent = `Only the top ${percentage}% tokens will be considered by the sampler.`;
  }
}

function setupSettings() {
  settingsButton.addEventListener("click", () => {
    settingsModal.style.display = "flex";
    updateSettingsDisplay(loadSettings());
  });
  closeSettings.addEventListener("click", () => {
    settingsModal.style.display = "none";
  });
  ["temperature", "top-p", "presence-penalty", "frequency-penalty"].forEach(id => {
    const element = document.getElementById(id);
    element.addEventListener("input", (e) => {
      const settings = loadSettings();
      const value = parseFloat(e.target.value);
      const key = id.replace(/-/g, '_');
      settings[key] = value;
      saveSettings(settings);
      updateSettingsDisplay(settings);
    });
  });
  settingsModal.addEventListener("mousedown", (e) => {
    if (e.target === settingsModal) {
      settingsModal.style.display = "none";
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      settingsModal.style.display = "none";
    }
  });
}

function setupCompletions() {
  completeButton.addEventListener("click", sendCompletion);
  completionsStopButton.addEventListener("click", stopCompletion);
  completionsSettingsButton.addEventListener("click", () => {
    settingsModal.style.display = "flex";
    updateSettingsDisplay(loadSettings());
  });
  completionsInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      sendCompletion();
    }
  });
}

function stopCompletion() {
  if (abortController) {
    abortController.abort();
    cleanupAfterCompletion();
  }
}

function cleanupAfterCompletion() {
  completeButton.style.display = "inline-block";
  completionsStopButton.style.display = "none";
  completeButton.disabled = false;
  abortController = null;

  // select newly added text and restore focus
  const textArea = completionsInput;
  textArea.focus();
  textArea.selectionStart = originalLength || 0;
  textArea.selectionEnd = textArea.value.length;
}

async function sendCompletion() {
  const text = completionsInput.value;
  completeButton.style.display = "none";
  completionsStopButton.style.display = "inline-block";
  completeButton.disabled = true;
  abortController = new AbortController();
  originalLength = text.length;
  completionsStopButton.focus();
  const settings = loadSettings();
  try {
    const response = await fetch(`${getUrlPrefix()}/v1/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${API_KEY}`
      },
      body: JSON.stringify({
        model: flagz.model || "gpt-3.5-turbo",
        prompt: text,
        temperature: settings.temperature,
        top_p: settings.top_p,
        presence_penalty: settings.presence_penalty,
        frequency_penalty: settings.frequency_penalty,
        stream: true
      }),
      signal: abortController.signal
    });
    if (response.ok) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done)
            break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          for (let i = 0; i < lines.length - 1; i++) {
            const line = lines[i].trim();
            if (line.startsWith("data: ")) {
              const data = line.slice(6);
              if (data === "[DONE]")
                continue;
              try {
                const parsed = JSON.parse(data);
                const content = parsed.choices[0]?.text || "";
                completionsInput.value += content;
                completionsInput.scrollTop = completionsInput.scrollHeight;
              } catch (e) {
                console.error("Error parsing JSON:", e);
              }
            }
          }
          buffer = lines[lines.length - 1];
        }
      } catch (error) {
        if (error.name !== "AbortError") {
          console.error("Error reading stream:", error);
        }
      }
    } else {
      console.error("Completion failed:", response.statusText);
    }
  } catch (error) {
    if (error.name !== "AbortError") {
      console.error("Completion error:", error);
    }
  } finally {
    cleanupAfterCompletion();
  }
}

function removeLastDirectChild(element) {
  if (element.lastElementChild) {
    element.removeChild(element.lastElementChild);
  }
}

function onRedo() {
  if (!chatHistory.length)
    return;
  removeLastDirectChild(chatMessages);
  let msg = chatHistory.pop();
  if (msg.role === "assistant") {
    removeLastDirectChild(chatMessages);
    msg = chatHistory.pop();
  }
  chatInput.value = msg.content;
  chatInput.focus();
  chatInput.dispatchEvent(new Event("input")); // adjust textarea height
}

function setupMenu() {
  const triggers = document.querySelectorAll('.menu-trigger');
  const menus = document.querySelectorAll('.menu');
  const chatModeSwitch = document.getElementById('chat-mode-switch');
  const completionsModeSwitch = document.getElementById('completions-mode-switch');
  if (flagz.is_base_model) {
    completionsModeSwitch.classList.add('disabled');
    completionsModeSwitch.title = "Chatbot mode isn't possible because this is a base model that hasn't had instruction fine tuning; try passing --chat-template chatml or llama2 if this is really an instruct model.";
    completionsModeSwitch.disabled = true;
  }
  triggers.forEach(trigger => {
    trigger.addEventListener('click', (e) => {
      e.stopPropagation();
      const menu = trigger.nextElementSibling;
      menus.forEach(m => {
        if (m !== menu)
          m.classList.remove('show');
      });
      menu.classList.toggle('show');
    });
  });
  document.addEventListener('click', () => {
    menus.forEach(menu => menu.classList.remove('show'));
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape')
      menus.forEach(menu => menu.classList.remove('show'));
  });
  chatModeSwitch.addEventListener('click', () => {
    flagz.completion_mode = true;
    setupCompletionsMode();
    menus.forEach(menu => menu.classList.remove('show'));
  });
  completionsModeSwitch.addEventListener('click', () => {
    if (!flagz.is_base_model) {
      flagz.completion_mode = false;
      setupChatCompletionsMode();
      menus.forEach(menu => menu.classList.remove('show'));
    }
  });
}

function setupChatCompletionsMode() {
  chatInterface.style.display = "flex";
  completionsInterface.style.display = "none";
  startChat([{ role: "system", content: getSystemPrompt() }]);
  chatInput.focus();
}

function setupCompletionsMode() {
  chatInterface.style.display = "none";
  completionsInterface.style.display = "flex";
  completionsInput.focus();
}

function onUploadButtonClick() {
  fileUpload.click();
}

function onFileUploadChange(e) {
  if (e.target.files[0]) {
    onFile(e.target.files[0]);
    e.target.value = '';
  }
}

async function chatbot() {
  flagz = await fetchFlagz();
  updateModelInfo();
  setupSettings();
  setupCompletions();
  setupMenu();
  if (flagz.is_base_model || flagz.completion_mode) {
    setupCompletionsMode();
  } else {
    setupChatCompletionsMode();
  }
  sendButton.addEventListener("click", sendMessage);
  stopButton.addEventListener("click", stopMessage);
  redoButton.addEventListener("click", onRedo);
  chatInput.addEventListener("input", onChatInput);
  chatInput.addEventListener("keydown", onKeyDown);
  chatMessages.addEventListener("touchmove", onWheel);
  document.addEventListener("wheel", onWheel);
  document.addEventListener("dragenter", onDragBegin);
  document.addEventListener("dragover", onDragBegin);
  document.addEventListener("dragleave", onDragEnd);
  document.addEventListener("drop", onDragEnd);
  document.addEventListener("drop", onDrop);
  document.addEventListener("paste", onPaste);
  uploadButton.addEventListener("click", onUploadButtonClick);
  fileUpload.addEventListener("change", onFileUploadChange);
}

chatbot();
