/**
 * Shiretto Cat: Edge AI Pipeline Logic
 */
import { pipeline, env } from '@xenova/transformers';

const CAFE_BEHAVIORS: Record<string, { pose: string, anchor: 'rim' | 'base' | 'center', offset: number }> = {
  'cup': { pose: 'peeking', anchor: 'rim', offset: 0.1 },
  'mug': { pose: 'peeking', anchor: 'rim', offset: 0.1 },
  'bottle': { pose: 'peeking', anchor: 'rim', offset: 0.15 },
  'wine glass': { pose: 'peeking', anchor: 'rim', offset: 0.05 },
  'plate': { pose: 'standing', anchor: 'rim', offset: 0.02 },
  'bowl': { pose: 'peeking', anchor: 'rim', offset: 0.1 },
  'sandwich': { pose: 'standing', anchor: 'rim', offset: -0.05 },
  'bread': { pose: 'standing', anchor: 'rim', offset: -0.05 },
  'cake': { pose: 'standing', anchor: 'rim', offset: -0.05 },
  'toast': { pose: 'standing', anchor: 'rim', offset: -0.05 },
  'chair': { pose: 'peeking', anchor: 'center', offset: 0.2 },
  'table': { pose: 'walking', anchor: 'base', offset: 0 },
  'default': { pose: 'walking', anchor: 'base', offset: 0 }
};

export class ShirettoPipeline {
  private segmenter: any = null;
  private isInitializing = false;

  async init() {
    if (this.segmenter || this.isInitializing) return;
    this.isInitializing = true;
    try {
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      const isWebGPUSupported = !!navigator.gpu;
      // 最も汎用性が高く、カフェの小物を捉えやすいモデルを固定
      this.segmenter = await pipeline('image-segmentation', 'Xenova/detr-resnet-50-panoptic', {
        device: isWebGPUSupported ? 'webgpu' : 'wasm',
        dtype: isWebGPUSupported ? 'fp16' : 'fp32'
      });
    } catch (error) {
      console.error('Core init failed:', error);
    } finally {
      this.isInitializing = false;
    }
  }

  async process(imageSource: string): Promise<any> {
    if (!this.segmenter) await this.init();
    try {
      const output = await this.segmenter(imageSource);
      const { allCleaned, analysis } = this.analyzeScene(output);
      const placement = this.solvePlacement(analysis);
      
      const result = await this.drawCatToDataURL(imageSource, placement);
      
      return {
        result,
        debugInfo: {
          labels: allCleaned.filter((s:any) => s.label !== 'object').map((s: any) => s.label),
          placement,
          analysis
        }
      };
    } catch (error) {
      console.error('Processing failed:', error);
      throw error;
    }
  }

  private async drawCatToDataURL(source: string, placement: any): Promise<string> {
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) return resolve(source);
        ctx.drawImage(img, 0, 0);
        this.drawCat(canvas, placement);
        resolve(canvas.toDataURL('image/jpeg', 0.9));
      };
      img.src = source;
    });
  }

  private analyzeScene(output: any[]) {
    const all = output.map((s: any) => {
      let label = s.label.replace(/LABEL_\d+,?\s*/gi, '').toLowerCase().split(',')[0].trim();
      // IDからの緊急マッピング
      if (!label && s.label.includes('187')) label = 'cup';
      if (!label && s.label.includes('149')) label = 'bottle';
      return { ...s, label: label || 'object', bounds: this.estimateBounds(s) };
    });

    // 「前景（下半分）にあるカフェアイテム」を最優先でスコアリング
    const candidates = all.filter(o => o.bounds.maxY > 0.4 && o.bounds.maxY < 0.95);
    const topItem = candidates.sort((a, b) => {
      const aScore = (CAFE_BEHAVIORS[a.label] ? 100 : 0) + (a.bounds.maxY * 50);
      const bScore = (CAFE_BEHAVIORS[b.label] ? 100 : 0) + (b.bounds.maxY * 50);
      return bScore - aScore;
    })[0] || all[0];

    return { analysis: { target: topItem } };
  }

  private solvePlacement(analysis: any) {
    const obj = analysis.target;
    if (!obj) return { x: 0.2, y: 0.8, scale: 0.12, rotation: -5, pose: 'walking', side: 'left' };

    const b = obj.bounds;
    const behavior = CAFE_BEHAVIORS[obj.label] || CAFE_BEHAVIORS['default'];
    
    let px = (b.minX + b.maxX) / 2;
    let py = b.minY; // 基本は上
    let side = 'right';

    if (behavior.anchor === 'rim') {
      // 縁にひっかける
      px = b.minX + (b.maxX - b.minX) * 0.7; // 少し右寄り
      py = b.minY + ( (b.maxY - b.minY) * behavior.offset);
    } else if (behavior.anchor === 'base') {
      // 地面に置く
      py = b.maxY;
    } else {
      // 中央付近
      py = b.minY + (b.maxY - b.minY) * behavior.offset;
    }

    // UI回避
    if (px > 0.7 && py > 0.7) px = 0.2;

    return {
      x: Math.max(0.05, Math.min(0.95, px)),
      y: Math.max(0.05, Math.min(0.95, py)),
      scale: Math.max(0.08, Math.min(0.2, (b.maxX - b.minX) * 0.8)),
      rotation: px > 0.5 ? 5 : -5,
      pose: behavior.pose,
      side: 'right',
      reason: `Found ${obj.label}. Applied ${behavior.pose} on ${behavior.anchor}.`
    };
  }

  private estimateBounds(segment: any) {
    const mask = segment.mask;
    let minX = mask.width, maxX = 0, minY = mask.height, maxY = 0;
    for (let i = 0; i < mask.data.length; i++) {
        if (mask.data[i] > 128) {
            const x = i % mask.width;
            const y = Math.floor(i / mask.width);
            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
        }
    }
    return {
        minX: minX / mask.width, maxX: maxX / mask.width,
        minY: minY / mask.height, maxY: maxY / mask.height
    };
  }

  drawCat(canvas: HTMLCanvasElement, placement: any) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const { x, y, scale, rotation, pose, side } = placement;
    const px = x * canvas.width;
    const py = y * canvas.height;
    const size = scale * canvas.width;

    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(rotation * Math.PI / 180);
    this.generateCatLines(ctx, size, pose, side);
    ctx.restore();
  }

  private generateCatLines(ctx: CanvasRenderingContext2D, size: number, pose: string, side: string) {
    ctx.strokeStyle = 'white';
    ctx.lineWidth = Math.max(1.5, size / 30); // さらに繊細に
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowBlur = 4;
    ctx.shadowColor = 'rgba(255, 255, 255, 0.5)';
    const dir = side === 'right' ? 1 : -1;
    ctx.scale(dir, 1);

    if (pose === 'peeking') {
      ctx.beginPath();
      // Face
      ctx.moveTo(-size * 0.4, 0);
      ctx.quadraticCurveTo(-size * 0.4, -size * 0.4, -size * 0.25, -size * 0.5);
      ctx.lineTo(-size * 0.3, -size * 0.7);
      ctx.lineTo(-size * 0.1, -size * 0.6);
      ctx.quadraticCurveTo(0, -size * 0.65, size * 0.1, -size * 0.6);
      ctx.lineTo(size * 0.3, -size * 0.7);
      ctx.lineTo(size * 0.25, -size * 0.5);
      ctx.quadraticCurveTo(size * 0.4, -size * 0.4, size * 0.4, 0);
      // Paws
      ctx.moveTo(-size * 0.45, 0); ctx.quadraticCurveTo(-size * 0.45, size * 0.05, -size * 0.35, size * 0.05); ctx.quadraticCurveTo(-size * 0.25, size * 0.05, -size * 0.25, 0);
      ctx.moveTo(size * 0.25, 0); ctx.quadraticCurveTo(size * 0.25, size * 0.05, size * 0.35, size * 0.05); ctx.quadraticCurveTo(size * 0.45, size * 0.05, size * 0.45, 0);
      ctx.stroke();
      // Eyes
      ctx.beginPath(); ctx.arc(-size * 0.12, -size * 0.35, size * 0.025, 0, Math.PI * 2); ctx.arc(size * 0.12, -size * 0.35, size * 0.025, 0, Math.PI * 2);
      ctx.fillStyle = 'white'; ctx.fill();
    } else if (pose === 'walking') {
      ctx.beginPath();
      ctx.moveTo(-size * 0.8, -size * 0.2); // 尻尾
      ctx.quadraticCurveTo(-size * 1.2, 0, -size * 0.8, size * 0.2);
      ctx.quadraticCurveTo(-size * 0.5, -size * 0.2, -size * 0.3, -size * 0.6); // 背中
      ctx.quadraticCurveTo(0, -size * 0.8, size * 0.2, -size * 1.0); // 首
      ctx.lineTo(size * 0.25, -size * 1.2); // 耳
      ctx.lineTo(size * 0.35, -size * 1.0);
      ctx.quadraticCurveTo(size * 0.5, -size * 0.8, size * 0.45, -size * 0.4); // 顔
      ctx.quadraticCurveTo(size * 0.35, 0, size * 0.3, -size * 0.1); // 前脚
      ctx.stroke();
    }
  }
}

export const pipelineInstance = new ShirettoPipeline();
