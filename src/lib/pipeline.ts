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
    
    // 実行環境の安定化設定（init内で実行）
    env.allowLocalModels = false;
    env.useBrowserCache = true;
    // @ts-ignore
    env.backends.onnx.wasm.proxy = false; 
    // @ts-ignore
    env.backends.onnx.wasm.numThreads = 1;

    // WebGPUのサポート確認
    // @ts-ignore
    const isWebGPUSupported = !!navigator.gpu;
    console.log('WebGPU Support:', isWebGPUSupported);

    try {
      // 最初から安全なフォールバックを考慮して初期化
      const device = isWebGPUSupported ? 'webgpu' : 'wasm';
      
      // @ts-ignore
      this.segmenter = await pipeline('image-segmentation', 'Xenova/slimsam-0.125-unified', {
        device: device,
      });
      console.log(`AI Pipeline initialized with ${device}`);
    } catch (error) {
      console.warn('Primary initialization failed, trying WASM fallback:', error);
      try {
        this.segmenter = await pipeline('image-segmentation', 'Xenova/slimsam-0.125-unified', {
          device: 'wasm',
        });
      } catch (wasmError) {
        console.error('All backends failed:', wasmError);
      }
    } finally {
      this.isInitializing = false;
    }
  }

  async process(imageSource: string): Promise<{ result: string, debugInfo: any }> {
    if (!this.segmenter) await this.init();

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
    const container = output.find((s: any) => ['cup', 'bowl', 'bottle', 'plate', 'mug'].includes(s.label));
    const table = output.find((s: any) => ['table', 'desk', 'surface', 'cloth'].includes(s.label));
    const contents = output.find((s: any) => ['liquid', 'food', 'coffee', 'tea'].includes(s.label));

    // 簡単な幾何学情報の抽出 (実際のピクセルデータから計算する代わりにモック情報を補強)
    // 実際には各segmentの mask プロパティを走査して計算する
    return {
      container: container ? { ...container, bounds: this.estimateBounds(container) } : null,
      table: table ? { ...table, bounds: this.estimateBounds(table) } : null,
      contents: contents ? { ...contents, bounds: this.estimateBounds(contents) } : null,
    };
  }

  private estimateBounds(segment: any) {
    // モック: 実際には segment.mask (ImageData) から最小/最大座標を算出する
    return {
      minX: 0.3, maxX: 0.7, minY: 0.4, maxY: 0.8 // 割合
    };
  }

  private calculatePlacement(analysis: any) {
    const { container, table } = analysis;

    if (!container) {
      return { x: 0.8, y: 0.8, scale: 0.15, rotation: 0, reason: 'No container found' };
    }

    // 基本戦略: 容器の下端（maxY）かつ左右どちらかの端に近いテーブル接地ポイント
    const x = container.bounds.maxX + 0.05; // 容器の右側
    const y = container.bounds.maxY - 0.05; // 容器の下の方（接地付近）

    return {
      x: Math.min(0.9, x),
      y: Math.min(0.9, y),
      scale: 0.12,
      rotation: -5,
      reason: 'Side of container found'
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

        // デバッグ用: 認識されたバウンディングボックスの描画
        if (analysis.container) {
          ctx.strokeStyle = 'rgba(0, 255, 0, 0.5)';
          ctx.strokeRect(
            analysis.container.bounds.minX * img.width,
            analysis.container.bounds.minY * img.height,
            (analysis.container.bounds.maxX - analysis.container.bounds.minX) * img.width,
            (analysis.container.bounds.maxY - analysis.container.bounds.minY) * img.height
          );
        }

        const catSize = img.width * placement.scale;
        const x = img.width * placement.x;
        const y = img.height * placement.y;

        // 猫の描画 (ミニマルな線画)
        ctx.strokeStyle = 'white';
        ctx.lineWidth = Math.max(2, img.width / 400);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.shadowBlur = 4;
        ctx.shadowColor = 'rgba(0,0,0,0.5)';

        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(placement.rotation * Math.PI / 180);
        
        ctx.beginPath();
        // 首から上を覗かせるようなポーズ
        ctx.moveTo(-catSize/2, catSize/2);
        ctx.quadraticCurveTo(-catSize/2.2, -catSize/2, -catSize/2.5, -catSize/1.2); // 左耳
        ctx.lineTo(-catSize/5, -catSize/1.8);
        ctx.lineTo(catSize/5, -catSize/1.8);
        ctx.lineTo(catSize/2.5, -catSize/1.2); // 右耳
        ctx.quadraticCurveTo(catSize/2.2, -catSize/2, catSize/2, catSize/2);
        ctx.stroke();

        // 閉じた目 (しれっとした表情)
        ctx.beginPath();
        ctx.moveTo(-catSize/4, -catSize/4);
        ctx.lineTo(-catSize/8, -catSize/4);
        ctx.moveTo(catSize/4, -catSize/4);
        ctx.lineTo(catSize/8, -catSize/4);
        ctx.stroke();

        ctx.restore();

        resolve(canvas.toDataURL('image/jpeg', 0.9));
      };
      img.src = source;
    });
  }
}

export const pipelineInstance = new ShirettoPipeline();
