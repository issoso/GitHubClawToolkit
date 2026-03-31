import { GoogleGenAI } from '@google/genai';
import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';

const TEMPLATE_ROOT = path.resolve(process.env.TEMPLATE_ROOT || process.cwd());

function normalizeText(value: unknown): string {
  return String(value ?? '')
    .replace(/\r\n/g, '\n')
    .replace(/\u00a0/g, ' ')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]{2,}/g, ' ')
    .trim();
}

function stripAnsi(text: unknown): string {
  return String(text ?? '').replace(/\u001b\[[0-9;?]*[ -/]*[@-~]/g, '');
}

async function readUserPrompt(promptFilePath: string): Promise<string> {
  const userPrompt = normalizeText(await readFile(promptFilePath, 'utf8'));
  if (!userPrompt) throw new Error(`${promptFilePath} 內容為空，無法摘要。`);
  return userPrompt;
}

async function resolveContextJsonlPath(): Promise<string> {
  const contextJsonlFile = process.env.CONTEXT_JSONL_FILE;
  if (contextJsonlFile) {
    const absPath = path.resolve(contextJsonlFile);
    await readFile(absPath, 'utf8');
    return absPath;
  }

  const files = await readdir(process.cwd(), { withFileTypes: true });
  const jsonlFile = files.find((entry) => entry.isFile() && entry.name.endsWith('.jsonl'));
  if (!jsonlFile) throw new Error('找不到可用的上下文 jsonl 檔案（根目錄 *.jsonl）。');
  return path.resolve(jsonlFile.name);
}

async function readContextJsonl(contextPath: string): Promise<string> {
  const raw = await readFile(contextPath, 'utf8');
  const normalized = stripAnsi(raw).replace(/\r\n/g, '\n').trim();
  if (!normalized) throw new Error(`${contextPath} 內容為空，無法提供上下文。`);
  return normalized;
}

async function readSystemPrompt(): Promise<string> {
  const systemPromptFile = process.env.SYSTEM_PROMPT_FILE
    ? path.resolve(process.env.SYSTEM_PROMPT_FILE)
    : path.join(TEMPLATE_ROOT, 'SYSTEM.md');
  const systemPrompt = normalizeText(await readFile(systemPromptFile, 'utf8'));
  if (!systemPrompt) throw new Error(`${systemPromptFile} 內容為空，無法摘要。`);
  return systemPrompt;
}

function buildPrompt(userPrompt: string, contextJsonl: string): string {
  return [
    '使用者需求（user.md）：',
    '"""',
    userPrompt,
    '"""',
    '',
    '上下文（jsonl）：',
    '"""',
    contextJsonl,
    '"""',
    '',
    '請根據使用者需求與上下文輸出摘要。',
    '若需要補充最新的公開資訊或查證內容，可以使用 Google Search。',
  ].join('\n');
}

async function main(): Promise<void> {
  const apiKey = normalizeText(process.env.VERTEXAI_SUMMARY_API_KEY || process.env.VERTEXAI_API_KEY);
  if (!apiKey) throw new Error('缺少 VERTEXAI_SUMMARY_API_KEY');

  const promptFile = process.env.PROMPT_FILE;
  if (!promptFile) throw new Error('缺少 PROMPT_FILE 環境變數');
  const model = normalizeText(process.env.VERTEXAI_SUMMARY_MODEL) || 'google/gemini-2.5-flash';

  const userPrompt = await readUserPrompt(promptFile);
  const contextJsonlPath = await resolveContextJsonlPath();
  const contextJsonl = await readContextJsonl(contextJsonlPath);
  const systemPrompt = await readSystemPrompt();

  const ai = new GoogleGenAI({
    vertexai: true,
    apiKey,
    apiVersion: 'v1',
  });

  const result = await ai.models.generateContent({
    model,
    config: {
      systemInstruction: systemPrompt,
      tools: [{ googleSearch: {} }],
    },
    contents: [{
      role: 'user',
      parts: [{
        text: buildPrompt(userPrompt, contextJsonl),
      }],
    }],
  });

  const summary = normalizeText(result.text);
  if (!summary) throw new Error('模型未回傳任何摘要內容。');

  console.log(summary);
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  process.exitCode = 1;
});
