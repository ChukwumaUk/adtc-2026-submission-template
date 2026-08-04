/* Cassava Advisor — front end.
   Streams one question's answer from POST /ask and drives the stage rail.
   No dependencies; everything here runs offline. */

(function () {
  "use strict";

  var form = document.getElementById("ask-form");
  var questionEl = document.getElementById("question");
  var askBtn = document.getElementById("ask-btn");
  var stopBtn = document.getElementById("stop-btn");
  var hintEl = document.getElementById("hint");
  var provenanceEl = document.getElementById("provenance");
  var examplesEl = document.getElementById("examples");
  var runEl = document.getElementById("run");
  var railEl = document.getElementById("rail");
  var statusEl = document.getElementById("status");
  var sourcesEl = document.getElementById("sources");
  var sourceListEl = document.getElementById("source-list");
  var answerEl = document.getElementById("answer");
  var answerTextEl = document.getElementById("answer-text");
  var cursorEl = document.getElementById("cursor");
  var warningEl = document.getElementById("warning");
  var warningBodyEl = document.getElementById("warning-body");
  var completionEl = document.getElementById("completion");
  var errorEl = document.getElementById("error");
  var errorBodyEl = document.getElementById("error-body");

  var steps = Array.prototype.slice.call(railEl.querySelectorAll(".step"));

  // The six guides in the store, named the way a reader would say them.
  var DOC_NAMES = {
    "iita-disease-control-cassava": "IITA — Cassava disease control",
    "iita-pest-control-cassava": "IITA — Cassava pest control",
    "iita-basics-cassava-varieties": "IITA BASICS — Improved cassava varieties",
    "fao-save-and-grow-cassava": "FAO — Save and Grow: Cassava",
    "ashc-cassava-cropping-guide": "ASHC — Cassava cropping guide",
    "naerls-cassava-production-processing": "NAERLS — Cassava production and processing"
  };

  var ACRONYMS = { iita: 1, fao: 1, ashc: 1, naerls: 1, basics: 1 };

  function documentName(slug) {
    if (DOC_NAMES[slug]) return DOC_NAMES[slug];
    var words = String(slug).split(/[-_]/).map(function (w) {
      return ACRONYMS[w] ? w.toUpperCase() : w;
    });
    var first = words.shift() || "";
    if (!ACRONYMS[first.toLowerCase()]) {
      first = first.charAt(0).toUpperCase() + first.slice(1);
    }
    return words.length ? first + " — " + words.join(" ") : first;
  }

  // --- Stage rail --------------------------------------------------------

  var controller = null;
  var ticker = null;
  var stepStartedAt = 0;
  var runStartedAt = 0;
  var activeStep = -1;
  var advanceTimer = null;

  // The spine is drawn between the first and last marker centres. Measuring
  // beats percentages, which overshoot when a step label wraps.
  var spine = { top: 0, height: 0 };

  function measureSpine() {
    var first = steps[0].querySelector(".step-mark");
    var last = steps[steps.length - 1].querySelector(".step-mark");
    var railTop = railEl.getBoundingClientRect().top;
    var a = first.getBoundingClientRect();
    var b = last.getBoundingClientRect();
    spine.top = a.top - railTop + a.height / 2;
    spine.height = (b.top + b.height / 2) - (a.top + a.height / 2);
    railEl.style.setProperty("--spine-top", spine.top + "px");
    railEl.style.setProperty("--spine-height", spine.height + "px");
  }

  function paintFill(fraction) {
    railEl.style.setProperty("--fill", spine.height * fraction + "px");
  }

  function setStep(index) {
    if (index <= activeStep) return;
    activeStep = index;
    stepStartedAt = Date.now();
    steps.forEach(function (step, i) {
      if (i < index) {
        step.setAttribute("data-state", "done");
      } else if (i === index) {
        step.setAttribute("data-state", "active");
      } else {
        step.removeAttribute("data-state");
      }
    });
    measureSpine();
    paintFill(index / (steps.length - 1));
    tick();
  }

  function completeSteps() {
    clearTimeout(advanceTimer);
    activeStep = steps.length;
    steps.forEach(function (step) { step.setAttribute("data-state", "done"); });
    measureSpine();
    paintFill(1);
    stopTicker();
  }

  function haltSteps() {
    // Used when a run is stopped or fails: no false claim of completion.
    clearTimeout(advanceTimer);
    steps.forEach(function (step) {
      if (step.getAttribute("data-state") === "active") {
        step.removeAttribute("data-state");
      }
    });
    stopTicker();
  }

  function tick() {
    steps.forEach(function (step, i) {
      var slot = step.querySelector(".step-time");
      if (i === activeStep) {
        slot.textContent = Math.round((Date.now() - stepStartedAt) / 1000) + "s";
      } else if (i > activeStep) {
        slot.textContent = "";
      }
    });
  }

  function startTicker() {
    stopTicker();
    ticker = setInterval(tick, 250);
  }

  function stopTicker() {
    if (ticker) { clearInterval(ticker); ticker = null; }
  }

  // --- SSE over fetch ----------------------------------------------------

  function parseEvent(block) {
    var name = "message";
    var data = "";
    block.split("\n").forEach(function (line) {
      if (line.indexOf("event:") === 0) name = line.slice(6).trim();
      else if (line.indexOf("data:") === 0) data += line.slice(5).trim();
    });
    var payload = {};
    if (data) {
      try { payload = JSON.parse(data); } catch (e) { payload = {}; }
    }
    return { name: name, payload: payload };
  }

  // --- Event handling ----------------------------------------------------

  var answerBuffer = "";
  var sawWarning = false;

  function handle(evt) {
    var d = evt.payload;

    if (evt.name === "stage") {
      if (d.stage === "retrieving") {
        setStep(0);
        statusEl.textContent = "Reading your question against the guides.";
        // Retrieval expands the wording, then searches. Both happen inside
        // the same second or two, so step two lights up shortly after.
        advanceTimer = setTimeout(function () { setStep(1); }, 900);
      } else if (d.stage === "generating") {
        setStep(2);
        statusEl.textContent = "Writing. This can take up to a minute.";
        answerEl.hidden = false;
        cursorEl.hidden = false;
      }
      return;
    }

    if (evt.name === "sources") {
      setStep(1);
      renderSources(d.sources || []);
      return;
    }

    if (evt.name === "token") {
      answerEl.hidden = false;
      answerBuffer += d.text || "";
      answerTextEl.textContent = answerBuffer;
      return;
    }

    if (evt.name === "warning") {
      sawWarning = true;
      warningBodyEl.textContent = d.message || "";
      warningEl.hidden = false;
      return;
    }

    if (evt.name === "done") {
      completeSteps();
      cursorEl.hidden = true;
      statusEl.textContent = "";
      var seconds = Math.round(d.elapsed || (Date.now() - runStartedAt) / 1000);
      if (d.grounded) {
        completionEl.textContent =
          "Advice complete. " + d.sources + " guide passages used, " +
          seconds + " seconds." +
          (sawWarning ? " Read the safety warning above before acting." : "");
      } else {
        completionEl.textContent =
          "Complete. No guidance close enough to answer this, " +
          seconds + " seconds.";
      }
      completionEl.hidden = false;
      return;
    }

    if (evt.name === "error") {
      haltSteps();
      cursorEl.hidden = true;
      statusEl.textContent = "";
      errorBodyEl.textContent = d.message || "The request failed.";
      errorEl.hidden = false;
      return;
    }
  }

  function renderSources(list) {
    // Several of the top passages usually come from the same guide. Merge
    // them so the reader sees documents, not repeats.
    var order = [];
    var byDoc = {};
    list.forEach(function (item) {
      var key = item.document;
      if (!byDoc[key]) { byDoc[key] = { passages: 0, score: 0 }; order.push(key); }
      byDoc[key].passages += 1;
      byDoc[key].score = Math.max(byDoc[key].score, item.score);
    });

    sourceListEl.textContent = "";
    order.forEach(function (key) {
      var info = byDoc[key];
      var li = document.createElement("li");
      var chip = document.createElement("span");
      chip.className = "chip chip-source";

      var doc = document.createElement("span");
      doc.className = "doc";
      doc.textContent = documentName(key) +
        (info.passages > 1 ? " (" + info.passages + " passages)" : "");

      var score = document.createElement("span");
      score.className = "score";
      score.textContent = Math.round(info.score * 100) + "% match";

      chip.appendChild(doc);
      chip.appendChild(score);
      li.appendChild(chip);
      sourceListEl.appendChild(li);
    });

    sourcesEl.hidden = list.length === 0;
  }

  // --- Run lifecycle -----------------------------------------------------

  function resetRun() {
    clearTimeout(advanceTimer);
    answerBuffer = "";
    sawWarning = false;
    activeStep = -1;
    answerTextEl.textContent = "";
    answerEl.hidden = true;
    cursorEl.hidden = true;
    warningEl.hidden = true;
    completionEl.hidden = true;
    errorEl.hidden = true;
    sourcesEl.hidden = true;
    sourceListEl.textContent = "";
    statusEl.textContent = "";
    steps.forEach(function (step) {
      step.removeAttribute("data-state");
      step.querySelector(".step-time").textContent = "";
    });
    railEl.style.setProperty("--fill", "0px");
  }

  function setBusy(busy) {
    askBtn.disabled = busy;
    askBtn.textContent = busy ? "Working" : "Get advice";
    stopBtn.hidden = !busy;
  }

  async function ask(question) {
    resetRun();
    examplesEl.hidden = true;
    runEl.hidden = false;
    setBusy(true);
    runStartedAt = Date.now();
    startTicker();
    setStep(0);

    controller = new AbortController();

    try {
      var response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question }),
        signal: controller.signal
      });

      if (!response.ok || !response.body) {
        throw new Error("The advisor is not responding (" + response.status + ").");
      }

      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";

      while (true) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        var split;
        while ((split = buffer.indexOf("\n\n")) !== -1) {
          var block = buffer.slice(0, split);
          buffer = buffer.slice(split + 2);
          if (block.trim()) handle(parseEvent(block));
        }
      }
    } catch (err) {
      if (err && err.name === "AbortError") {
        haltSteps();
        cursorEl.hidden = true;
        statusEl.textContent = "";
        completionEl.textContent = answerBuffer
          ? "Stopped. The advice above is incomplete."
          : "Stopped before any advice was written.";
        completionEl.hidden = false;
      } else {
        haltSteps();
        cursorEl.hidden = true;
        statusEl.textContent = "";
        errorBodyEl.textContent = err && err.message
          ? err.message
          : "The connection to the advisor was lost.";
        errorEl.hidden = false;
      }
    } finally {
      stopTicker();
      setBusy(false);
      controller = null;
    }
  }

  // --- Wiring ------------------------------------------------------------

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    var question = questionEl.value.trim();
    if (question) ask(question);
  });

  stopBtn.addEventListener("click", function () {
    if (controller) controller.abort();
  });

  window.addEventListener("resize", function () {
    if (activeStep < 0 || runEl.hidden) return;
    measureSpine();
    paintFill(Math.min(activeStep / (steps.length - 1), 1));
  });

  // Enter sends, Shift+Enter makes a new line.
  questionEl.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  Array.prototype.forEach.call(
    document.querySelectorAll(".chip-example"),
    function (chip) {
      chip.addEventListener("click", function () {
        // The language tag is a label, not part of the question.
        var tag = chip.querySelector(".chip-tag");
        var text = tag
          ? chip.textContent.replace(tag.textContent, "").trim()
          : chip.textContent.trim();
        questionEl.value = text;
        ask(text);
      });
    }
  );

  fetch("/api/status")
    .then(function (r) { return r.json(); })
    .then(function (s) {
      if (s.passages) {
        provenanceEl.textContent =
          "Works offline. " + s.passages + " passages from six vetted field guides.";
      }
      if (!s.model_ready) {
        hintEl.textContent =
          "The language model is not running. Start llama-server on " +
          s.llama_server + " before asking.";
      }
    })
    .catch(function () { /* status is a courtesy; the ask flow reports its own errors */ });
})();
