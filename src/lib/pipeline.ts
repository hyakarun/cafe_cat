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

      const isWebGPUSupported = !!navigator.gpu;
      console.log('WebGPU Support:', isWebGPUSupported);

      try {
        const device = isWebGPUSupported ? 'webgpu' : 'wasm';
        this.segmenter = await pipeline('image-segmentation', 'Xenova/detr-resnet-50-panoptic', { device });
        console.log(`AI Pipeline initialized with ${device}`);
      } catch (err) {
        console.warn('Failed with primary device, falling back to wasm...', err);
        // Fallback specifically to WASM if WebGPU acts up
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

    // 1. Perform Segmentation
    const output = await this.segmenter(imageSource);
    console.log('AI Segmentation Raw Output:', output);

    // 2. Analyze Masks (Step A)
    const analysis = this.analyzeSegments(output);
    
    // 3. Determine Placement (Step B & C)
    const placement = this.calculatePlacement(analysis);

    // 4. Synthesize
    const result = await this.drawCat(imageSource, placement, analysis);
    
    return {
      result,
      debugInfo: {
        labels: output.map(s => s.label),
        placement,
        analysis
      }
    };
  }

  private analyzeSegments(output: any[]) {
    // 容器、テーブル、中身などの領域を整理
    const container = output.find((s: any) => {
      const l = s.label.toLowerCase();
      return ['cup', 'bowl', 'bottle', 'plate', 'mug', 'coffee', 'glass'].some(kw => l.includes(kw));
    });
    const table = output.find((s: any) => ['table', 'desk', 'surface', 'cloth'].some(kw => s.label.toLowerCase().includes(kw)));
    
    // 見つからなかった場合は、面積がマズマズ大きいオブジェクトをフォールバックにする（一旦1つ目の結果）
    const targetObj = container || (output.length > 0 ? output[0] : null);

    return {
      container: targetObj ? { ...targetObj, bounds: this.estimateBounds(targetObj) } : null,
      table: table ? { ...table, bounds: this.estimateBounds(table) } : null,
    };
  }

  private estimateBounds(segment: any) {
    if (!segment || !segment.mask) {
      return { minX: 0.3, maxX: 0.7, minY: 0.4, maxY: 0.8 };
    }
    
    try {
      const mask = segment.mask;
      const width = mask.width;
      const height = mask.height;
      const data = mask.data;
      
      if (!width || !height || !data || data.length === 0) {
        return { minX: 0.3, maxX: 0.7, minY: 0.4, maxY: 0.8 };
      }

      let minX = width, maxX = 0, minY = height, maxY = 0;
      let found = false;
      const channels = mask.channels || (data.length / (width * height));
      
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = (y * width + x) * channels;
          // RGBAならアルファ値(idx+3)、グレースケールならそのまま(idx)を使用
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
      
      if (!found) {
        return { minX: 0.3, maxX: 0.7, minY: 0.4, maxY: 0.8 };
      }
      
      return { 
        minX: minX / width, 
        maxX: maxX / width, 
        minY: minY / height, 
        maxY: maxY / height 
      };
    } catch (err) {
      console.warn("Bounds estimation failed:", err);
      return { minX: 0.3, maxX: 0.7, minY: 0.4, maxY: 0.8 };
    }
  }

  private calculatePlacement(analysis: any) {
    const { container, table } = analysis;

    if (!container || !container.mask) {
      return { x: 0.8, y: 0.8, scale: 0.15, rotation: 10, reason: 'AI could not find a clear object' };
    }

    // AI解析: 境界点（エッジ）の抽出
    const edges = this.extractGroundingEdges(container, table);
    
    // スコアリングによって「最も自然（しれっと）な場所」を選択
    const bestPoint = edges.length > 0 ? edges[0] : { x: container.bounds.maxX, y: container.bounds.maxY };
    
    // シーンの文脈（容器の大きさや位置）から猫のスタイルを決定
    const boxWidth = container.bounds.maxX - container.bounds.minX;
    const boxHeight = container.bounds.maxY - container.bounds.minY;
    const isTall = boxHeight > boxWidth;
    
    // 猫のポーズ案を生成 (覗き込み / 佇む / 隠れる)
    const pose = isTall ? 'peeking' : 'sitting';
    const side = bestPoint.x > (container.bounds.maxX + container.bounds.minX) / 2 ? 'right' : 'left';

    return {
      x: bestPoint.x,
      y: bestPoint.y,
      scale: Math.max(0.12, Math.min(0.25, boxWidth * 1.2)),
      rotation: side === 'right' ? 15 : -15,
      pose,
      side,
      reason: `AI suggested ${side} side of the ${container.label} as a natural hiding spot.`
    };
  }

  /**
   * マスクデータを精査し、容器がテーブルと接しているエッジポイントを抽出する
   */
  private extractGroundingEdges(container: any, table: any) {
    const mask = container.mask;
    const data = mask.data;
    const width = mask.width;
    const height = mask.height;
    const channels = mask.channels || (data.length / (width * height));
    
    const candidates: {x: number, y: number, score: number}[] = [];

    // 容器の下端 20% の領域を重点的に走査
    const startY = Math.floor(container.bounds.minY * height + (container.bounds.maxY - container.bounds.minY) * height * 0.7);
    const endY = Math.floor(container.bounds.maxY * height);

    for (let y = startY; y < endY; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * channels;
        const alpha = channels === 4 ? data[idx + 3] : data[idx];
        
        if (alpha > 128) {
          // 容器の左右端を検出
          const isEdge = x < (container.bounds.minX * width + 5) || x > (container.bounds.maxX * width - 5);
          if (isEdge) {
            candidates.push({ 
              x: x / width, 
              y: y / height, 
              score: y // 下にあるほどスコアが高い（接地面に近い）
            });
          }
        }
      }
    }

    return candidates.sort((a, b) => b.score - a.score);
  }

  /**
   * AIの提案に基づき、その場に最適化されたミニマルな猫を生成（描画）する
   */
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

        // クリーンな白い線画の猫を生成的に描画
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
    
    // わずかに外光が反射しているようなグロー効果
    ctx.shadowBlur = 10;
    ctx.shadowColor = 'rgba(255, 255, 255, 0.3)';

    const dir = side === 'right' ? 1 : -1;
    
    ctx.beginPath();
    
    if (pose === 'peeking') {
      // 覗き込みポーズの生成
      // 耳
      ctx.moveTo(-size * 0.3 * dir, -size * 0.5);
      ctx.lineTo(-size * 0.4 * dir, -size * 0.9);
      ctx.lineTo(-size * 0.1 * dir, -size * 0.6);
      
      // 頭のライン
      ctx.quadraticCurveTo(size * 0.2 * dir, -size * 0.8, size * 0.5 * dir, -size * 0.5);
      
      // 反対の耳
      ctx.lineTo(size * 0.6 * dir, -size * 0.8);
      ctx.lineTo(size * 0.7 * dir, -size * 0.4);
      
      // 顔の輪郭
      ctx.quadraticCurveTo(size * 0.6 * dir, 0, 0, 0);
      
      // 閉じた目（しれっと感）
      ctx.stroke();
      
      ctx.beginPath();
      ctx.lineWidth = Math.max(2, size / 25);
      ctx.moveTo(size * 0.1 * dir, -size * 0.4);
      ctx.lineTo(size * 0.25 * dir, -size * 0.4);
      ctx.moveTo(size * 0.45 * dir, -size * 0.35);
      ctx.lineTo(size * 0.55 * dir, -size * 0.35);
    } else {
      // 佇むポーズの生成
      // 背中
      ctx.moveTo(0, 0);
      ctx.quadraticCurveTo(-size * 0.4 * dir, -size * 0.2, -size * 0.5 * dir, -size * 0.8);
      // 頭
      ctx.quadraticCurveTo(-size * 0.5 * dir, -size * 1.1, -size * 0.2 * dir, -size * 1.2);
      // 耳
      ctx.lineTo(-size * 0.3 * dir, -size * 1.4);
      ctx.lineTo(-size * 0.1 * dir, -size * 1.25);
      ctx.lineTo(size * 0.1 * dir, -size * 1.4);
      ctx.lineTo(size * 0.2 * dir, -size * 1.2);
      // 前面
      // しっぽ
      ctx.beginPath();
      ctx.moveTo(-size * 0.4 * dir, -size * 0.2);
      ctx.quadraticCurveTo(-size * 0.8 * dir, -size * 0.1, -size * 0.7 * dir, -size * 0.4);
      ctx.stroke();
    }
  }
}

export const pipelineInstance = new ShirettoPipeline();
