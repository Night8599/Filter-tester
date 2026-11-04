// Talks to filter_tester.web_api.
//
// - In the browser (GitHub Pages): runs the Python package with Pyodide.
// - With ?backend=server: sends the same calls to web/dev_server.py, which
//   runs the package with regular Python (useful for development and tests).
//
// Both expose: await backend.call(functionName, ...args) -> parsed JSON (or a number).

export const PYODIDE_VERSION = "0.26.4";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

class ServerBackend {
  constructor() {
    this.kind = "server";
  }

  async init(onStatus) {
    onStatus("Connecting to the local Python server…");
    await this.call("scenarios");
  }

  async call(fn, ...args) {
    const response = await fetch("api", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fn, args }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
    return body.result;
  }
}

class PyodideBackend {
  constructor() {
    this.kind = "pyodide";
  }

  async init(onStatus) {
    onStatus("Downloading the Python runtime (first visit can take a few seconds)…");
    await loadScript(`${PYODIDE_URL}pyodide.js`);
    this.pyodide = await globalThis.loadPyodide({ indexURL: PYODIDE_URL });
    onStatus("Loading numpy and scipy…");
    await this.pyodide.loadPackage(["numpy", "scipy"]);
    onStatus("Loading filter_tester…");
    const response = await fetch("filter_tester.zip", { cache: "no-cache" });
    if (!response.ok) throw new Error(`could not download filter_tester.zip (HTTP ${response.status})`);
    this.pyodide.unpackArchive(await response.arrayBuffer(), "zip", { extractDir: "/home/pyodide/pkg" });
    this.pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide/pkg')");
    this.api = this.pyodide.pyimport("filter_tester.web_api");
  }

  async call(fn, ...args) {
    const result = this.api[fn](...args);
    return typeof result === "string" ? JSON.parse(result) : result;
  }
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`could not load ${src}`));
    document.head.appendChild(script);
  });
}

export function createBackend() {
  const params = new URLSearchParams(location.search);
  return params.get("backend") === "server" ? new ServerBackend() : new PyodideBackend();
}
