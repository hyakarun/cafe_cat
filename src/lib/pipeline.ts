/**
 * Shiretto Cat: Edge AI Pipeline Logic
 */
import { pipeline, env } from '@xenova/transformers';

export class ShirettoPipeline {
  private segmenter: any = null;
  private isInitializing: boolean = false;

  async init() {
    if (this.segmenter || this.isInitializing) return;
    this.isInitializing = true;
    
    try {
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      // @ts-ignore
      if (env.backends && env.backends.onnx && env.backends.onnx.wasm) {
        env.backends.onnx.wasm.proxy = false; 
        env.backends.onnx.wasm.numThreads = 1;
      }

      // @ts-ignore
      const isWebGPUSupported = !!navigator.gpu;
      console.log('WebGPU Support:', isWebGPUSupported);

      try {
        const device = isWebGPUSupported ? 'webgpu' : 'wasm';
        // @ts-ignore
        this.segmenter = await pipeline('image-segmentation', 'Xenova/detr-resnet-50-panoptic', { device });
        console.log(`AI Pipeline initialized with ${device}`);
      } catch (err) {
        console.warn('Failed with primary device, falling back to wasm...', err);
        // @ts-ignore
        this.segmenter = await pipeline('image-segmentation', 'Xenova/detr-resnet-50-panoptic', { device: 'wasm' });
        console.log(`AI Pipeline initialized with wasm via fallback`);
      }
    } catch (error) {
      console.error('Initialization completely failed:', error);
      throw error;
    } finally {
      this.isInitializing = false;
    }
  }

  async process(imageSource: string): Promise<{ result: string, debugInfo: any }> {
    if (!this.segmenter) await this.init();
    if (!this.segmenter) throw new Error("AI segmenter failed to initialize properly.");

    const output = await this.segmenter(imageSource);
    const { cleanedOutput, analysis } = this.analyzeSegments(output);
    const placement = this.calculatePlacement(analysis);
    const result = await this.drawCat(imageSource, placement, analysis);
    
    return {
      result,
      debugInfo: {
        labels: cleanedOutput.map((s: any) => s.cleanLabel),
        placement,
        analysis
      }
    };
  }

  private analyzeSegments(output: any[]) {
    const CAFE_TARGETS = [
      'cup', 'mug', 'glass', 'bottle', 'beverage', 'drink',
      'plate', 'bowl', 'dish', 'spoon', 'fork', 'knife',
      'food', 'cake', 'sandwich', 'pizza', 'donut', 'fruit', 'bread'
    ];

    const cleanedOutput = output.map((s: any) => {
      let clean = s.label.replace(/LABEL_\d+,?\s*/gi, '').trim();
      clean = clean.split(',')[0].trim();
      return { ...s, cleanLabel: clean || 'Object' };
    });

    const validTargets = cleanedOutput.filter((s: any) => {
      const l = s.cleanLabel.toLowerCase();
      return CAFE_TARGETS.some(kw => l.includes(kw));
    });

    const targetObj = validTargets.length > 0 ? validTargets[0] : null;
    const tableObj = cleanedOutput.find((s: any) => s.cleanLabel.toLowerCase().includes('table'));

    return {
      cleanedOutput: validTargets,
      analysis: {
        container: targetObj ? { ...targetObj, label: targetObj.cleanLabel, bounds: this.estimateBounds(targetObj) } : null,
        table: tableObj ? { ...tableObj, label: tableObj.cleanLabel, bounds: this.estimateBounds(tableObj) } : null,
      }
    };
  }

  private estimateBounds(segment: any) {
    if (!segment || !segment.mask) return { minX: 0.3, maxX: 0.7, minY: 0.4, maxY: 0.8 };
    try {
      const mask = segment.mask;
      const { width, height, data } = mask;
      if (!width || !height || !data || data.length === 0) return { minX: 0.3, maxX: 0.7, minY: 0.4, maxY: 0.8 };

      let minX = width, maxX = 0, minY = height, maxY = 0;
      let found = false;
      const channels = mask.channels || (data.length / (width * height));
      
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = (y * width + x) * channels;
          const val = channels === 4 ? data[idx + 3] : data[idx];
          if (val > 128 || (channels === 1 && val > 0)) {
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            found = true;
          }
        }
      }
      if (!found) return { minX: 0.3, maxX: 0.7, minY: 0.4, maxY: 0.8 };
      return { minX: minX / width, maxX: maxX / width, minY: minY / height, maxY: maxY / height };
    } catch (err) {
      return { minX: 0.3, maxX: 0.7, minY: 0.4, maxY: 0.8 };
    }
  }

  private calculatePlacement(analysis: any) {
    const { container } = analysis;
    if (!container) return { x: 0.8, y: 0.8, scale: 0.15, rotation: 10, reason: 'No cafe items found' };

    const b = container.bounds;
    const centerX = (b.minX + b.maxX) / 2;
    const width = b.maxX - b.minX;
    const height = b.maxY - b.minY;
    const isVertical = height > width;

    let px, py, pose;
    if (isVertical) {
      px = centerX + (width * 0.45); 
      py = b.maxY - (height * 0.15);
      pose = 'peeking';
    } else {
      px = b.maxX - (width * 0.1);
      py = b.maxY + (height * 0.05);
      pose = 'sitting';
    }

    const pm = Math.max(0.8, Math.min(1.5, py * 1.5));
    const side = px > centerX ? 'right' : 'left';

    return {
      x: Math.min(0.9, Math.max(0.1, px)),
      y: Math.min(0.9, Math.max(0.1, py)),
      scale: Math.max(0.1, Math.min(0.25, (width * 0.8) * pm)),
      rotation: side === 'right' ? 12 : -12,
      pose,
      side,
      reason: `AI recognized a ${isVertical ? 'tall' : 'flat'} ${container.label}.`
    };
  }

  private async drawCat(source: string, placement: any, analysis: any): Promise<string> {
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
        const { x, y, scale, rotation, pose, side } = placement;
        const px = x * canvas.width;
        const py = y * canvas.height;
        const size = scale * canvas.width;

        ctx.save();
        ctx.translate(px, py);
        ctx.rotate(rotation * Math.PI / 180);
        this.generateCatLines(ctx, size, pose, side);
        ctx.restore();
        resolve(canvas.toDataURL('image/jpeg', 0.9));
      };
      img.src = source;
    });
  }

  private generateCatLines(ctx: CanvasRenderingContext2D, size: number, pose: string, side: string) {
    ctx.strokeStyle = 'white';
    ctx.lineWidth = Math.max(3, size / 15);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowBlur = 10;
    ctx.shadowColor = 'rgba(255, 255, 255, 0.3)';
    const dir = side === 'right' ? 1 : -1;
    ctx.beginPath();
    
    if (pose === 'peeking') {
      ctx.moveTo(-size * 0.3 * dir, -size * 0.5);
      ctx.lineTo(-size * 0.4 * dir, -size * 0.9);
      ctx.lineTo(-size * 0.1 * dir, -size * 0.6);
      ctx.quadraticCurveTo(size * 0.2 * dir, -size * 0.8, size * 0.5 * dir, -size * 0.5);
      ctx.lineTo(size * 0.6 * dir, -size * 0.8);
      ctx.lineTo(size * 0.7 * dir, -size * 0.4);
      ctx.quadraticCurveTo(size * 0.6 * dir, 0, 0, 0);
      ctx.stroke();
      ctx.beginPath();
      ctx.lineWidth = Math.max(2, size / 25);
      ctx.moveTo(size * 0.1 * dir, -size * 0.4);
      ctx.lineTo(size * 0.25 * dir, -size * 0.4);
      ctx.moveTo(size * 0.45 * dir, -size * 0.35);
      ctx.lineTo(size * 0.55 * dir, -size * 0.35);
    } else {
      ctx.moveTo(0, 0);
      ctx.quadraticCurveTo(-size * 0.4 * dir, -size * 0.2, -size * 0.5 * dir, -size * 0.8);
      ctx.quadraticCurveTo(-size * 0.5 * dir, -size * 1.1, -size * 0.2 * dir, -size * 1.2);
      ctx.lineTo(-size * 0.3 * dir, -size * 1.4);
      ctx.lineTo(-size * 0.1 * dir, -size * 1.25);
      ctx.lineTo(size * 0.1 * dir, -size * 1.4);
      ctx.lineTo(size * 0.2 * dir, -size * 1.2);
      ctx.quadraticCurveTo(size * 0.4 * dir, -size * 1.0, size * 0.3 * dir, 0);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(-size * 0.4 * dir, -size * 0.2);
      ctx.quadraticCurveTo(-size * 0.8 * dir, -size * 0.1, -size * 0.7 * dir, -size * 0.4);
    }
    ctx.stroke();
  }
}

export const pipelineInstance = new ShirettoPipeline();
