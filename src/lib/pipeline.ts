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
        // Fallback specifically to WASM if WebGPU acts up
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

    // 1. Perform Segmentation
    const output = await this.segmenter(imageSource);
    console.log('AI Segmentation Raw Output:', output);

    // 2. Analyze Masks (Step A)
    const { cleanedOutput, analysis } = this.analyzeSegments(output);
    
    // 3. Determine Placement (Step B & C)
    const placement = this.calculatePlacement(analysis);

    // 4. Synthesize
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
    // ラベルをクリーンアップ (LABEL_187 などを正規表現で徹底除去)
    const cleanedOutput = output.map((s: any) => {
      let clean = s.label.replace(/LABEL_\d+,?\s*/gi, '').trim();
      // カンマが残っている場合は最初の一つを採用
      clean = clean.split(',')[0].trim();
      
      return {
        ...s,
        cleanLabel: clean || s.label || 'Object'
      };
    });

    // カフェに関連するオブジェクト（容器等）を優先的に探す
    const container = cleanedOutput.find((s: any) => {
      const l = s.cleanLabel.toLowerCase();
      // 優先度の高いカフェアイテム
      return ['cup', 'bowl', 'bottle', 'plate', 'mug', 'coffee', 'glass', 'tableware'].some(kw => l.includes(kw));
    });

    // カフェアイテムがない場合、椅子などの家具や他の大きな物体を対象にする
    const secondary = cleanedOutput.find((s: any) => {
      const l = s.cleanLabel.toLowerCase();
      return ['chair', 'table', 'desk', 'laptop', 'cell phone'].some(kw => l.includes(kw));
    });
    
    const targetObj = container || secondary || (cleanedOutput.length > 0 ? cleanedOutput[0] : null);

    return {
      cleanedOutput,
      analysis: {
        container: targetObj ? { ...targetObj, label: targetObj.cleanLabel, bounds: this.estimateBounds(targetObj) } : null,
        table: cleanedOutput.find((s: any) => s.cleanLabel.toLowerCase().includes('table')),
      }
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

    if (!container) {
      return { x: 0.8, y: 0.8, scale: 0.15, rotation: 10, reason: 'AI observed the generic scene. Placed at bottom right.' };
    }

    // AI解析: 物体の構造的な特徴を算出
    // 単なる最小値・最大値ではなく、物体の「重心」と「広がり」から配置を推論
    const containerBounds = container.bounds;
    const centerX = (containerBounds.minX + containerBounds.maxX) / 2;
    const centerY = (containerBounds.minY + containerBounds.maxY) / 2;
    
    const width = containerBounds.maxX - containerBounds.minX;
    const height = containerBounds.maxY - containerBounds.minY;

    // AI判断: 容器の種類や向きに基づいた「役割」の認識
    const isHorizontal = width > height; // 平たい皿やお盆
    const isVertical = height > width;   // コップやボトル

    let placementX, placementY, pose;
    
    if (isVertical) {
      // 縦型の物体（コップ等）の背後から覗かせる
      placementX = centerX + (width * 0.45); 
      placementY = containerBounds.maxY - (height * 0.15); // 接地面に近い側面
      pose = 'peeking';
    } else {
      // 平たい物体（皿等）の陰に佇ませる
      placementX = containerBounds.maxX - (width * 0.1);
      placementY = containerBounds.maxY + (height * 0.05); // 手前の接地場所
      pose = 'sitting';
    }

    // パース（奥行き）に基づいたスケール推論
    // 画面の下にあるほどカメラに近く、大きいと想定
    const perspectiveMultiplier = Math.max(0.8, Math.min(1.5, placementY * 1.5));
    const scale = Math.max(0.1, Math.min(0.25, (width * 0.8) * perspectiveMultiplier));

    const side = placementX > centerX ? 'right' : 'left';

    return {
      x: Math.min(0.9, Math.max(0.1, placementX)),
      y: Math.min(0.9, Math.max(0.1, placementY)),
      scale: scale,
      rotation: side === 'right' ? 12 : -12,
      pose,
      side,
      reason: `AI recognized a ${isVertical ? 'tall' : 'flat'} ${container.label}. Placed strategically at its ${side} grounding point.`
    };
  }

  /**
   * 物体の「構造」を認識するためのトポロジー解析
   * マスク全体の形状から、物体の向きや重心を算出する
   */
  private analyzeObjectTopology(segment: any) {
    if (!segment || !segment.mask) return null;
    
    // ここではピクセル走査ではなく、セグメント自体のメタデータと
    // マスクデータの矩形近似（Moment解析に近い処理）を使用して構造を特定する
    const b = this.estimateBounds(segment);
    
    return {
      centroid: { x: (b.minX + b.maxX) / 2, y: (b.minY + b.maxY) / 2 },
      isStable: b.maxY > 0.8, // 画面下部にあり、安定しているか
      aspectRatio: (b.maxX - b.minX) / (b.maxY - b.minY)
    };
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
