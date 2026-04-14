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
    const { container } = analysis;

    if (!container) {
      return { x: 0.8, y: 0.8, scale: 0.15, rotation: 10, reason: 'No object found' };
    }

    const b = container.bounds;
    const boxWidth = b.maxX - b.minX;
    const boxHeight = b.maxY - b.minY;
    
    // 容器の右側、かつ下から30%くらい上（テーブルとの接地感）に配置
    const x = b.maxX; 
    const y = b.maxY - (boxHeight * 0.3);
    
    // スケールは被写体のサイズに比例させる
    const scale = Math.max(0.1, Math.min(0.2, boxWidth * 0.6));
    
    return {
      // 画面外にはみ出ないように 10% ～ 90% の間に収める
      x: Math.min(0.9, Math.max(0.1, x)),
      y: Math.min(0.9, Math.max(0.1, y)),
      scale: scale,
      rotation: 15, // しれっと顔を出す傾き
      reason: `Peeking from ${container.label}'s right edge`
    };
  }

  private async loadAndProcessCatAsset(): Promise<HTMLCanvasElement> {
    return new Promise((resolve) => {
      const img = new Image();
      img.src = '/cat_asset.png';
      img.onload = () => {
        const c = document.createElement('canvas');
        c.width = img.width;
        c.height = img.height;
        const ctx = c.getContext('2d', { willReadFrequently: true });
        if (!ctx) return resolve(c);
        
        ctx.drawImage(img, 0, 0);
        const imgData = ctx.getImageData(0, 0, c.width, c.height);
        const data = imgData.data;
        
        // #00FF00 (ネオングリーン) をクロマキー処理で透過
        for (let i = 0; i < data.length; i += 4) {
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];
          
          if (g > 200 && r < 50 && b < 50) {
            data[i + 3] = 0; // 完全に透過
          } else if (g > 150 && r < 100 && b < 100) {
            // エッジのアンチエイリアス処理
            data[i + 3] = Math.max(0, data[i + 3] - (g - 150));
          }
        }
        ctx.putImageData(imgData, 0, 0);
        resolve(c);
      };
      img.onerror = () => {
        console.warn("Cat asset not found or failed to load");
        const c = document.createElement('canvas'); // 空のキャンバス
        resolve(c);
      };
    });
  }

  private async drawCat(source: string, placement: any, analysis: any): Promise<string> {
    // まずAI生成の猫アセットをロード＆透過処理
    const catCanvas = await this.loadAndProcessCatAsset();

    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) return resolve(source);

        // 1. 元の写真をベースとして描画
        ctx.drawImage(img, 0, 0);

        // 2. 猫のサイズと位置を計算
        // 画像は手描き線より大きいので、スケールを少し強めに乗算します
        const catSize = img.width * placement.scale * 1.8;
        const x = img.width * placement.x;
        const y = img.height * placement.y;

        ctx.save();
        ctx.translate(x, y);
        // しれっと覗くような少しの傾き
        ctx.rotate(placement.rotation * Math.PI / 180);

        // 自然に見せるためのドロップシャドウ
        ctx.shadowBlur = 15;
        ctx.shadowColor = 'rgba(0,0,0,0.4)';
        ctx.shadowOffsetY = 5;

        if (catCanvas.width > 0) {
          // 元画像の縦横比を維持しながら描画
          const aspect = catCanvas.height / catCanvas.width;
          const h = catSize * aspect;
          // (x, y) が猫の中心(足元ではなく胴の中心)になるようにオフセット
          ctx.drawImage(catCanvas, -catSize / 2, -h * 0.4, catSize, h);
        }

        ctx.restore();
        resolve(canvas.toDataURL('image/jpeg', 0.9));
      };
      img.src = source;
    });
  }
}

export const pipelineInstance = new ShirettoPipeline();
