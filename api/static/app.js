// Cache the main DOM nodes used by the chat surface and developer panel
const feed = document.getElementById("message-feed");
const form = document.getElementById("query-form");
const input = document.getElementById("query-input");
const clearButton = document.getElementById("clear-chat");
const refreshStatusButton = document.getElementById("refresh-status");
const statusGrid = document.getElementById("status-grid");
const developerPanel = document.getElementById("developer-panel");
const developerModeToggle = document.getElementById("developer-mode-toggle");
const template = document.getElementById("message-template");

// Keep small pieces of UI state across refreshes where that helps the user
const welcomeMarkup = feed.innerHTML;
let developerMode = localStorage.getItem("t2d_developer_mode") === "true";

// Keep repeated DOM rendering logic consistent and safely escaped
function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function createTag(label) {
  const tag = document.createElement("span");
  tag.className = "message-tag";
  tag.textContent = label;
  return tag;
}

// Render user and platform messages with the shared chat card template
function appendMessage(role, bodyHtml, tags = [], extraClass = "") {
  const fragment = template.content.cloneNode(true);
  const card = fragment.querySelector(".message-card");
  const roleNode = fragment.querySelector(".message-role");
  const bodyNode = fragment.querySelector(".message-body");
  const tagCluster = fragment.querySelector(".tag-cluster");

  roleNode.textContent = role;
  bodyNode.innerHTML = bodyHtml;
  if (extraClass) {
    card.classList.add(extraClass);
  }
  tags.forEach((tagText) => tagCluster.appendChild(createTag(tagText)));

  feed.appendChild(fragment);
  feed.scrollTop = feed.scrollHeight;
}

// Lock the composer while a request is in flight so users cannot double submit
function setLoadingState(active) {
  const submitButton = document.getElementById("submit-query");
  submitButton.disabled = active;
  submitButton.textContent = active ? "Running..." : "Run query";
}

// Render backend status cards only when developer mode is enabled
function renderStatusCard(title, state, headline, details = []) {
  const card = document.createElement("div");
  card.className = "status-card";
  const chipState = state === "ok" ? "ok" : state === "warn" ? "warn" : "neutral";

  const detailMarkup = details.length
    ? `<p class="status-copy">${details.map(escapeHtml).join("<br />")}</p>`
    : "";

  card.innerHTML = `
    <div class="panel-head">
      <h3>${escapeHtml(title)}</h3>
      <span class="status-chip" data-state="${chipState}">${escapeHtml(headline)}</span>
    </div>
    ${detailMarkup}
  `;
  return card;
}

// Load backend status only on demand so the default user view stays clean
async function loadStatus() {
  if (!developerMode) {
    statusGrid.innerHTML = "";
    return;
  }
  statusGrid.innerHTML = `
    <div class="status-card skeleton"></div>
    <div class="status-card skeleton"></div>
    <div class="status-card skeleton"></div>
    <div class="status-card skeleton"></div>
  `;

  try {
    const response = await fetch("/backend-status");
    const payload = await response.json();
    statusGrid.innerHTML = "";

    statusGrid.appendChild(
      renderStatusCard(
        "SQLite",
        payload.sqlite.available ? "ok" : "warn",
        payload.sqlite.available ? "Available" : "Missing",
        [`Path: ${payload.sqlite.path}`]
      )
    );

    statusGrid.appendChild(
      renderStatusCard(
        "MongoDB",
        payload.mongodb.available ? "ok" : "warn",
        payload.mongodb.available ? "Connected" : payload.mongodb.backend,
        [`Records: ${payload.mongodb.sample_collection_count}`, `Configured: ${payload.mongodb.configured}`]
      )
    );

    statusGrid.appendChild(
      renderStatusCard(
        "Neo4j",
        payload.neo4j.available ? "ok" : "warn",
        payload.neo4j.available ? "Connected" : payload.neo4j.backend,
        [`Entities: ${payload.neo4j.entity_count}`, `Relationships: ${payload.neo4j.relation_count}`]
      )
    );

    statusGrid.appendChild(
      renderStatusCard(
        "Retrieval",
        payload.fallback_files.retrieval_available ? "ok" : "warn",
        payload.fallback_files.retrieval_backend,
        [
          `Manifest: ${payload.fallback_files.retrieval_manifest}`,
          payload.fallback_files.embedding_model
            ? `Embedding: ${payload.fallback_files.embedding_model}`
            : "Embedding: lexical baseline",
        ]
      )
    );
  } catch (error) {
    statusGrid.innerHTML = "";
    statusGrid.appendChild(
      renderStatusCard("Runtime status", "warn", "Unavailable", [error instanceof Error ? error.message : String(error)])
    );
  }
}

// Render the final answer with citations caveats and debug data kept together
function renderResponse(payload) {
  const citations = (payload.citations || [])
    .map(
      (citation) =>
        `<li><strong>${escapeHtml(citation.reference_id)}</strong>: ${escapeHtml(citation.title)} (${escapeHtml(
          citation.source
        )})</li>`
    )
    .join("");

  const caveats = (payload.caveats || [])
    .map((note) => `<li>${escapeHtml(note)}</li>`)
    .join("");

  const metadataCards = [
    ["Trace ID", payload.trace_id],
    ["Scope", payload.metadata?.scope_family],
    ["Route reason", payload.metadata?.route_reason],
    ["Synthesis", payload.metadata?.synthesis_mode],
  ]
    .filter(([, value]) => value)
    .map(
      ([label, value]) => `
        <div class="meta-card">
          <span>${escapeHtml(label)}</span>
          <div>${escapeHtml(value)}</div>
        </div>
      `
    )
    .join("");

  const debugBlock = developerMode
    ? `
      <div class="developer-block">
        ${metadataCards ? `<div class="meta-grid">${metadataCards}</div>` : ""}
        <details>
          <summary>Raw response</summary>
          <pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>
        </details>
      </div>
    `
    : "";

  const body = `
    <p class="answer-copy">${escapeHtml(payload.answer)}</p>
    ${citations ? `<h3>Citations</h3><ul>${citations}</ul>` : ""}
    ${caveats ? `<h3>Notes</h3><ul>${caveats}</ul>` : ""}
    ${debugBlock}
  `;

  const tags = developerMode
    ? [payload.question_class, payload.metadata?.question_class_name, payload.metadata?.routing_mode].filter(Boolean)
    : [];

  appendMessage("Platform", body, tags, "assistant-card");
}

// Reuse the same placeholder card while streamed updates arrive
function renderStreamingCard(card, text) {
  const body = card.querySelector(".message-body");
  if (!body) {
    return;
  }
  body.innerHTML = `<p class="answer-copy">${escapeHtml(text || "Retrieving evidence and assembling answer")}</p>`;
  feed.scrollTop = feed.scrollHeight;
}

// Send queries over SSE first and fall back to JSON when streaming is unavailable
async function runQuery(query) {
  appendMessage("You", `<p>${escapeHtml(query)}</p>`, ["Query"], "user-card");
  appendMessage("Platform", `<p class="loading-dots">Retrieving evidence and assembling answer</p>`, ["Working"], "assistant-card");

  const loadingCard = feed.lastElementChild;
  setLoadingState(true);

  try {
    if (typeof EventSource !== "undefined") {
      await new Promise((resolve) => {
        const eventSource = new EventSource(`/query/stream?query=${encodeURIComponent(query)}`);
        let streamedAnswer = "";

        eventSource.addEventListener("status", () => {
          renderStreamingCard(loadingCard, streamedAnswer || "Retrieving evidence and assembling answer");
        });

        eventSource.addEventListener("delta", (event) => {
          const payload = JSON.parse(event.data);
          streamedAnswer += payload.text || "";
          renderStreamingCard(loadingCard, streamedAnswer);
        });

        eventSource.addEventListener("final", (event) => {
          const payload = JSON.parse(event.data);
          loadingCard.remove();
          renderResponse(payload);
          eventSource.close();
          resolve();
        });

        eventSource.addEventListener("error", (event) => {
          let message = "The query failed.";
          if (event?.data) {
            try {
              message = JSON.parse(event.data).error || message;
            } catch (parseError) {
              message = message;
            }
          }
          loadingCard.remove();
          appendMessage("Platform", `<p>${escapeHtml(message)}</p>`, ["Error"], "assistant-card error-card");
          eventSource.close();
          resolve();
        });
      });
      return;
    }

    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const payload = await response.json();
    loadingCard.remove();
    if (!response.ok) {
      appendMessage(
        "Platform",
        `<p>${escapeHtml(payload.error || "The query failed.")}</p>`,
        ["Error"],
        "assistant-card error-card"
      );
      return;
    }
    renderResponse(payload);
  } catch (error) {
    loadingCard.remove();
    appendMessage(
      "Platform",
      `<p>${escapeHtml(error instanceof Error ? error.message : String(error))}</p>`,
      ["Error"],
      "assistant-card error-card"
    );
  } finally {
    setLoadingState(false);
  }
}

// Form submission and keyboard handling
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) {
    input.focus();
    return;
  }
  input.value = "";
  await runQuery(query);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

// Chat reset and developer status refresh controls
clearButton.addEventListener("click", () => {
  feed.innerHTML = welcomeMarkup;
  input.focus();
});

refreshStatusButton.addEventListener("click", () => {
  loadStatus();
});

// Suggested prompt shortcuts seed the composer with representative questions
document.querySelectorAll(".prompt-chip").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt || "";
    input.focus();
  });
});

// Developer mode toggles extra runtime metadata without changing the core UI
function applyDeveloperMode() {
  developerPanel.hidden = !developerMode;
  developerModeToggle.checked = developerMode;
}

developerModeToggle.addEventListener("change", () => {
  developerMode = developerModeToggle.checked;
  localStorage.setItem("t2d_developer_mode", String(developerMode));
  applyDeveloperMode();
  loadStatus();
});

// Initial UI state
applyDeveloperMode();
if (developerMode) {
  loadStatus();
}
