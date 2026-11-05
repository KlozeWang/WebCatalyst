import { readFile, writeFile } from "node:fs/promises";
import { PNG } from "pngjs";
import type { ActionExecutionMetadata, SimulationAction } from "./types.js";

export async function createActionOverlay(args: {
  beforeScreenshotPath: string;
  outputPath: string;
  action: SimulationAction;
  metadata?: ActionExecutionMetadata;
}): Promise<string | undefined> {
  const point = args.metadata?.hitTarget?.pagePoint;
  if (!point || (args.action.type !== "click" && args.action.type !== "hover")) return undefined;

  const png = await readFile(args.beforeScreenshotPath);
  const image = PNG.sync.read(png);
  drawOverlay(image, Math.round(point.x), Math.round(point.y));
  await writeFile(args.outputPath, PNG.sync.write(image));
  return args.outputPath;
}

function drawOverlay(image: PNG, x: number, y: number): void {
  drawLine(image, x, y - 32, x, y + 32, [255, 43, 43, 255], 3);
  drawLine(image, x - 32, y, x + 32, y, [255, 43, 43, 255], 3);
  drawCircle(image, x, y, 20, [255, 255, 255, 255], 5);
  drawCircle(image, x, y, 17, [255, 43, 43, 255], 3);
}

function drawCircle(image: PNG, cx: number, cy: number, radius: number, color: Color, thickness: number): void {
  const inner = Math.max(0, radius - thickness);
  const outerSq = radius * radius;
  const innerSq = inner * inner;
  for (let y = cy - radius; y <= cy + radius; y++) {
    for (let x = cx - radius; x <= cx + radius; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const d = dx * dx + dy * dy;
      if (d <= outerSq && d >= innerSq) setPixel(image, x, y, color);
    }
  }
}

function drawLine(image: PNG, x0: number, y0: number, x1: number, y1: number, color: Color, thickness: number): void {
  const dx = Math.abs(x1 - x0);
  const sx = x0 < x1 ? 1 : -1;
  const dy = -Math.abs(y1 - y0);
  const sy = y0 < y1 ? 1 : -1;
  let err = dx + dy;
  let x = x0;
  let y = y0;

  while (true) {
    fillRect(image, x - Math.floor(thickness / 2), y - Math.floor(thickness / 2), thickness, thickness, color);
    if (x === x1 && y === y1) break;
    const e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y += sy;
    }
  }
}

function fillRect(image: PNG, x: number, y: number, width: number, height: number, color: Color): void {
  for (let yy = y; yy < y + height; yy++) {
    for (let xx = x; xx < x + width; xx++) setPixel(image, xx, yy, color);
  }
}

type Color = [number, number, number, number];

function setPixel(image: PNG, x: number, y: number, color: Color): void {
  if (x < 0 || y < 0 || x >= image.width || y >= image.height) return;
  const index = (image.width * y + x) << 2;
  const alpha = color[3] / 255;
  const invAlpha = 1 - alpha;
  image.data[index] = Math.round(color[0] * alpha + image.data[index] * invAlpha);
  image.data[index + 1] = Math.round(color[1] * alpha + image.data[index + 1] * invAlpha);
  image.data[index + 2] = Math.round(color[2] * alpha + image.data[index + 2] * invAlpha);
  image.data[index + 3] = 255;
}
