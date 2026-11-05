import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { callJsonJudge } from "../src/simulation/judge.js";

const disableTemperatureEnv = "AGENS_SIM_JUDGE_DISABLE_TEMPERATURE";

async function captureRequestBody(disableTemperature: string | undefined): Promise<Record<string, unknown>> {
  const savedEnv = process.env[disableTemperatureEnv];
  const savedFetch = globalThis.fetch;
  let requestBody: Record<string, unknown> | undefined;

  try {
    if (disableTemperature === undefined) delete process.env[disableTemperatureEnv];
    else process.env[disableTemperatureEnv] = disableTemperature;

    globalThis.fetch = (async (_input, init) => {
      requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return new Response(JSON.stringify({
        model: "response-model",
        choices: [{ message: { content: "{}" } }],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;

    await callJsonJudge({
      provider: { baseUrl: "http://judge.test/v1", apiKey: "key" },
      model: "judge-model",
      systemPrompt: "Return JSON.",
      textParts: ["test"],
      attachments: [],
      progressLabel: "test",
    });
  } finally {
    globalThis.fetch = savedFetch;
    if (savedEnv === undefined) delete process.env[disableTemperatureEnv];
    else process.env[disableTemperatureEnv] = savedEnv;
  }

  assert.ok(requestBody);
  return requestBody;
}

test("simulation judge sends temperature by default", async () => {
  const requestBody = await captureRequestBody(undefined);
  assert.equal(requestBody.temperature, 0.2);
});

test("simulation judge omits temperature when disabled by environment", async () => {
  const requestBody = await captureRequestBody("true");
  assert.equal("temperature" in requestBody, false);
});

test("weighted routing uses the additional key and always sends Chat Completions", async () => {
  const savedFetch = globalThis.fetch;
  const savedRandom = Math.random;
  let requestUrl = "";
  let requestHeaders: HeadersInit | undefined;
  let requestBody: Record<string, unknown> | undefined;
  try {
    Math.random = () => 0.749;
    globalThis.fetch = (async (input, init) => {
      requestUrl = String(input);
      requestHeaders = init?.headers;
      requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return new Response(JSON.stringify({
        model: "response-model",
        choices: [{ message: { content: "{}" } }],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;

    const result = await callJsonJudge({
      provider: {
        baseUrl: "http://primary.test/v1",
        apiKey: "primary-key",
        wireApi: "responses",
      },
      routing: {
        additionalProvider: {
          baseUrl: "http://additional.test/v1",
          apiKey: "additional-key",
          wireApi: "responses",
        },
        ratio: 3,
      },
      model: "judge-model",
      systemPrompt: "Return JSON.",
      textParts: ["test"],
      attachments: [],
      progressLabel: "test",
    });

    assert.equal(requestUrl, "http://additional.test/v1/chat/completions");
    assert.equal(new Headers(requestHeaders).get("Authorization"), "Bearer additional-key");
    assert.ok(requestBody && "messages" in requestBody);
    assert.equal(result.provenance.baseUrl, "http://additional.test/v1");
  } finally {
    globalThis.fetch = savedFetch;
    Math.random = savedRandom;
  }
});

test("routing boundary keeps the primary endpoint and its Responses format", async () => {
  const savedFetch = globalThis.fetch;
  const savedRandom = Math.random;
  let requestUrl = "";
  let requestHeaders: HeadersInit | undefined;
  let requestBody: Record<string, unknown> | undefined;
  try {
    Math.random = () => 0.75;
    globalThis.fetch = (async (input, init) => {
      requestUrl = String(input);
      requestHeaders = init?.headers;
      requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return new Response(JSON.stringify({
        model: "response-model",
        output: [{ content: [{ type: "output_text", text: "{}" }] }],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;

    await callJsonJudge({
      provider: {
        baseUrl: "http://primary.test/v1",
        apiKey: "primary-key",
        wireApi: "responses",
      },
      routing: {
        additionalProvider: {
          baseUrl: "http://additional.test/v1",
          apiKey: "additional-key",
        },
        ratio: 3,
      },
      model: "judge-model",
      systemPrompt: "Return JSON.",
      textParts: ["test"],
      attachments: [],
      progressLabel: "test",
    });

    assert.equal(requestUrl, "http://primary.test/v1/responses");
    assert.equal(new Headers(requestHeaders).get("Authorization"), "Bearer primary-key");
    assert.ok(requestBody && "input" in requestBody);
  } finally {
    globalThis.fetch = savedFetch;
    Math.random = savedRandom;
  }
});

test("HTTP 402, 403, and 429 retries stay on the initially selected endpoint", async () => {
  const savedFetch = globalThis.fetch;
  const savedRandom = Math.random;
  const savedSetTimeout = globalThis.setTimeout;
  let randomCalls = 0;
  try {
    Math.random = () => {
      randomCalls += 1;
      return 0;
    };
    globalThis.setTimeout = ((callback: () => void) => {
      callback();
      return 0 as unknown as ReturnType<typeof setTimeout>;
    }) as typeof setTimeout;

    for (const status of [402, 403, 429]) {
      const requestUrls: string[] = [];
      globalThis.fetch = (async (input) => {
        requestUrls.push(String(input));
        if (requestUrls.length === 1) return new Response("retryable error", { status });
        return new Response(JSON.stringify({
          model: "response-model",
          choices: [{ message: { content: "{}" } }],
        }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }) as typeof fetch;

      await callJsonJudge({
        provider: { baseUrl: "http://primary.test/v1", apiKey: "primary-key" },
        routing: {
          additionalProvider: { baseUrl: "http://additional.test/v1", apiKey: "additional-key" },
          ratio: 1,
        },
        model: "judge-model",
        systemPrompt: "Return JSON.",
        textParts: ["test"],
        attachments: [],
        progressLabel: "test",
      });

      assert.deepEqual(requestUrls, [
        "http://additional.test/v1/chat/completions",
        "http://additional.test/v1/chat/completions",
      ]);
    }
    assert.equal(randomCalls, 3);
  } finally {
    globalThis.fetch = savedFetch;
    Math.random = savedRandom;
    globalThis.setTimeout = savedSetTimeout;
  }
});

test("retryable HTTP errors stop after two retries", async () => {
  const savedFetch = globalThis.fetch;
  const savedSetTimeout = globalThis.setTimeout;
  let requests = 0;
  try {
    globalThis.setTimeout = ((callback: () => void) => {
      callback();
      return 0 as unknown as ReturnType<typeof setTimeout>;
    }) as typeof setTimeout;
    globalThis.fetch = (async () => {
      requests += 1;
      return new Response("forbidden", { status: 403 });
    }) as typeof fetch;

    await assert.rejects(
      callJsonJudge({
        provider: { baseUrl: "http://judge.test/v1", apiKey: "key" },
        model: "judge-model",
        systemPrompt: "Return JSON.",
        textParts: ["test"],
        attachments: [],
        progressLabel: "test",
      }),
      /judge returned 403/,
    );
    assert.equal(requests, 3);
  } finally {
    globalThis.fetch = savedFetch;
    globalThis.setTimeout = savedSetTimeout;
  }
});

test("fetch failures retry twice and can recover", async () => {
  const savedFetch = globalThis.fetch;
  const savedSetTimeout = globalThis.setTimeout;
  let requests = 0;
  try {
    globalThis.setTimeout = ((callback: () => void) => {
      callback();
      return 0 as unknown as ReturnType<typeof setTimeout>;
    }) as typeof setTimeout;
    globalThis.fetch = (async () => {
      requests += 1;
      if (requests < 3) throw new TypeError("fetch failed");
      return new Response(JSON.stringify({
        model: "response-model",
        choices: [{ message: { content: "{}" } }],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;

    const result = await callJsonJudge({
      provider: { baseUrl: "http://judge.test/v1", apiKey: "key" },
      model: "judge-model",
      systemPrompt: "Return JSON.",
      textParts: ["test"],
      attachments: [],
      progressLabel: "test",
    });

    assert.deepEqual(result.parsed, {});
    assert.equal(requests, 3);
  } finally {
    globalThis.fetch = savedFetch;
    globalThis.setTimeout = savedSetTimeout;
  }
});

test("fetch failures stop after two retries", async () => {
  const savedFetch = globalThis.fetch;
  const savedSetTimeout = globalThis.setTimeout;
  let requests = 0;
  try {
    globalThis.setTimeout = ((callback: () => void) => {
      callback();
      return 0 as unknown as ReturnType<typeof setTimeout>;
    }) as typeof setTimeout;
    globalThis.fetch = (async () => {
      requests += 1;
      throw new TypeError("fetch failed");
    }) as typeof fetch;

    await assert.rejects(
      callJsonJudge({
        provider: { baseUrl: "http://judge.test/v1", apiKey: "key" },
        model: "judge-model",
        systemPrompt: "Return JSON.",
        textParts: ["test"],
        attachments: [],
        progressLabel: "test",
      }),
      /fetch failed/,
    );
    assert.equal(requests, 3);
  } finally {
    globalThis.fetch = savedFetch;
    globalThis.setTimeout = savedSetTimeout;
  }
});

test("HTTP 400 with an image retries once using compressed image data", async () => {
  const tempDir = await mkdtemp(path.join(tmpdir(), "agens-judge-compression-"));
  const imagePath = path.join(tempDir, "screenshot.png");
  await writeFile(imagePath, Buffer.from("original-image"));
  const savedFetch = globalThis.fetch;
  const requestBodies: Array<Record<string, any>> = [];
  let compressionCalls = 0;
  try {
    globalThis.fetch = (async (_input, init) => {
      requestBodies.push(JSON.parse(String(init?.body)) as Record<string, any>);
      if (requestBodies.length === 1) return new Response("invalid body", { status: 400 });
      return new Response(JSON.stringify({
        model: "response-model",
        choices: [{ message: { content: "{}" } }],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;

    await callJsonJudge({
      provider: { baseUrl: "http://judge.test/v1", apiKey: "key" },
      model: "judge-model",
      systemPrompt: "Return JSON.",
      textParts: [],
      attachments: [{ kind: "image", path: imagePath, mimeType: "image/png" }],
      progressLabel: "test",
      imageCompressor: async (inputPath) => {
        compressionCalls += 1;
        assert.equal(inputPath, imagePath);
        return { data: Buffer.from("compressed-image"), mimeType: "image/jpeg" };
      },
    });

    assert.equal(requestBodies.length, 2);
    assert.equal(compressionCalls, 1);
    const firstImage = requestBodies[0].messages[1].content[0].image_url.url as string;
    const secondImage = requestBodies[1].messages[1].content[0].image_url.url as string;
    assert.match(firstImage, /^data:image\/png;base64,/);
    assert.match(secondImage, /^data:image\/jpeg;base64,/);
    assert.notEqual(firstImage, secondImage);
  } finally {
    globalThis.fetch = savedFetch;
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("successful image requests do not invoke compression", async () => {
  const tempDir = await mkdtemp(path.join(tmpdir(), "agens-judge-no-compression-"));
  const imagePath = path.join(tempDir, "screenshot.png");
  await writeFile(imagePath, Buffer.from("original-image"));
  const savedFetch = globalThis.fetch;
  let compressionCalls = 0;
  try {
    globalThis.fetch = (async () => new Response(JSON.stringify({
      model: "response-model",
      choices: [{ message: { content: "{}" } }],
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })) as typeof fetch;

    await callJsonJudge({
      provider: { baseUrl: "http://judge.test/v1", apiKey: "key" },
      model: "judge-model",
      systemPrompt: "Return JSON.",
      textParts: [],
      attachments: [{ kind: "image", path: imagePath, mimeType: "image/png" }],
      progressLabel: "test",
      imageCompressor: async () => {
        compressionCalls += 1;
        return { data: Buffer.from("compressed-image"), mimeType: "image/jpeg" };
      },
    });

    assert.equal(compressionCalls, 0);
  } finally {
    globalThis.fetch = savedFetch;
    await rm(tempDir, { recursive: true, force: true });
  }
});
