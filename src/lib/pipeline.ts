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
      // 認証情報の干渉を防ぐためのクリーン設定
      env.allowLocalModels = false;
      env.useBrowserCache = false; // 古いキャッシュが壊れている可能性を考慮して一旦OFF
      
      // キャッシュストレージのクリア（可能であれば）
      try {
        if ('caches' in window) {
          await caches.delete('transformers-cache').catch(() => {});
        }
      } catch (e) {}

      // @ts-ignore
      const isWebGPUSupported = !!navigator.gpu;
      
      try {
        console.log(`Force loading AI engine...`);
        // 推論精度の高い Mask2Former を、認証に干渉されにくい形式でリトライ
        // @ts-ignore
        this.segmenter = await pipeline('image-segmentation', 'Xenova/detr-resnet-50-panoptic', {
          // @ts-ignore
          device: isWebGPUSupported ? 'webgpu' : 'wasm'
        });
        console.log('AI Pipeline force-loaded successfully.');
      } catch (err) {
        console.warn('AI loading failed twice. This is likely a network/auth block.', err);
      }
    } catch (error) {
      console.error('AI initialization failed:', error);
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
    // カフェに関連するオブジェクトの検索範囲を大幅に拡張
    const CAFE_TARGETS = [
      'cup', 'mug', 'glass', 'bottle', 'beverage', 'drink', 'latte', 'coffee', 'paper cup', 'plastic cup',
      'plate', 'bowl', 'dish', 'spoon', 'fork', 'knife', 'tray', 'tableware',
      'food', 'cake', 'sandwich', 'pizza', 'donut', 'fruit', 'bread', 'snack', 'toast', 'dessert', 'muffin'
    ];

    // すべてのオブジェクトをデバッグ用に保持
    const allCleaned = output.map((s: any) => {
      let clean = s.label.replace(/LABEL_\d+,?\s*/gi, '').trim();
      clean = clean.split(',')[0].trim();
      return { ...s, cleanLabel: clean || 'Object' };
    });

    // ホワイトリストに該当するものを優先ターゲット候補に
    const targetCandidates = allCleaned.filter((s: any) => {
      const l = s.cleanLabel.toLowerCase();
      return CAFE_TARGETS.some(kw => l.includes(kw));
    });

    const targetObj = targetCandidates.length > 0 ? targetCandidates[0] : null;

    // フォールバック: 中央付近にある最も大きな物体
    const fallbackObj = !targetObj && allCleaned.length > 0 
      ? allCleaned.sort((a, b) => {
          const ab = this.estimateBounds(a);
          const bb = this.estimateBounds(b);
          return (bb.maxX - bb.minX) * (bb.maxY - bb.minY) - (ab.maxX - ab.minX) * (ab.maxY - ab.minY);
        })[0]
      : null;

    const finalTarget = targetObj || fallbackObj;
    const tableObj = allCleaned.find((s: any) => s.cleanLabel.toLowerCase().includes('table'));

    return {
      cleanedOutput: allCleaned, // すべて表示して何が起きてるか見せる
      analysis: {
        container: finalTarget ? { ...finalTarget, label: finalTarget.cleanLabel, bounds: this.estimateBounds(finalTarget) } : null,
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
    // デフォルト: 見つからない場合は中央左寄りの影にひっそり
    if (!container) return { x: 0.2, y: 0.6, scale: 0.15, rotation: -5, pose: 'walking', side: 'left', reason: 'Shadow search' };

    const b = container.bounds;
    const centerX = (b.minX + b.maxX) / 2;
    const width = b.maxX - b.minX;
    const height = b.maxY - b.minY;
    const isVertical = height > width;

    let px, py, pose, side;

    if (isVertical) {
      // コップなどの縦長：上端から「覗き」
      px = centerX + (width * 0.1); 
      py = b.minY + (height * 0.05); // 上端に近い位置
      pose = 'peeking';
      side = 'right';
    } else {
      // 皿・パンなどの平坦：その上に「佇む」 or 足元で「歩く」
      if (Math.random() > 0.5) {
        px = b.minX + (width * 0.3);
        py = b.minY - (height * 0.1); // 物体の上に少し浮く（足が着く位置）
        pose = 'standing';
        side = 'right';
      } else {
        px = b.maxX + 0.05;
        py = b.maxY;
        pose = 'walking';
        side = 'left';
      }
    }

    const pm = Math.max(0.7, Math.min(1.2, py));
    
    // UI回避
    let finalX = Math.min(0.9, Math.max(0.1, px));
    let finalY = Math.min(0.9, Math.max(0.1, py));
    if (finalX > 0.7 && finalY > 0.7) { 
      finalX = 0.2; finalY = b.maxY; pose = 'walking';
    }

    return {
      x: finalX,
      y: finalY,
      scale: Math.max(0.08, Math.min(0.18, (width * 0.6) * pm)),
      rotation: finalX > 0.5 ? 8 : -8,
      pose,
      side,
      reason: `AI detected ${container.label}. Placed at ${pose === 'peeking' ? 'top rim' : 'surface'}.`
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
    ctx.lineWidth = Math.max(2, size / 25); // 線を細めに、より洗練された印象に
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowBlur = 6;
    ctx.shadowColor = 'rgba(255, 255, 255, 0.4)';
    
    const dir = side === 'right' ? 1 : -1;
    ctx.scale(dir, 1); // 向きを反転

    if (pose === 'peeking') {
      // 「覗き (Peek)」の再現
      ctx.beginPath();
      // 顔（輪郭）
      ctx.moveTo(-size * 0.4, 0);
      ctx.quadraticCurveTo(-size * 0.4, -size * 0.4, -size * 0.25, -size * 0.5); // 左頬
      ctx.lineTo(-size * 0.3, -size * 0.7); // 左耳外側
      ctx.lineTo(-size * 0.1, -size * 0.6); // 左耳内側
      ctx.quadraticCurveTo(0, -size * 0.65, size * 0.1, -size * 0.6); // 頭頂部
      ctx.lineTo(size * 0.3, -size * 0.7); // 右耳内側
      ctx.lineTo(size * 0.25, -size * 0.5); // 右耳外側
      ctx.quadraticCurveTo(size * 0.4, -size * 0.4, size * 0.4, 0); // 右頬
      
      // 手 (Paws)
      // 左手
      ctx.moveTo(-size * 0.45, 0);
      ctx.quadraticCurveTo(-size * 0.45, size * 0.05, -size * 0.35, size * 0.05);
      ctx.quadraticCurveTo(-size * 0.25, size * 0.05, -size * 0.25, 0);
      // 右手
      ctx.moveTo(size * 0.25, 0);
      ctx.quadraticCurveTo(size * 0.25, size * 0.05, size * 0.35, size * 0.05);
      ctx.quadraticCurveTo(size * 0.45, size * 0.05, size * 0.45, 0);
      ctx.stroke();

      // 目 (Eyes) - しれっとした表情
      ctx.beginPath();
      ctx.arc(-size * 0.12, -size * 0.35, size * 0.03, 0, Math.PI * 2);
      ctx.arc(size * 0.12, -size * 0.35, size * 0.03, 0, Math.PI * 2);
      ctx.fillStyle = 'white';
      ctx.fill();
    } else if (pose === 'walking') {
      // 「歩く (Walk)」の再現
      ctx.beginPath();
      ctx.moveTo(-size * 0.8, -size * 0.3); // 尻尾の先
      ctx.quadraticCurveTo(-size * 1.0, -size * 0.1, -size * 0.7, -size * 0.15); // 尻尾
      ctx.quadraticCurveTo(-size * 0.4, -size * 0.5, -size * 0.2, -size * 0.5); // 背中
      ctx.quadraticCurveTo(size * 0.1, -size * 0.6, size * 0.3, -size * 0.9); // 首〜頭
      ctx.lineTo(size * 0.35, -size * 1.0); // 耳
      ctx.lineTo(size * 0.45, -size * 0.9);
      ctx.quadraticCurveTo(size * 0.6, -size * 0.7, size * 0.5, -size * 0.5); // 顔
      ctx.quadraticCurveTo(size * 0.3, -size * 0.3, size * 0.3, -size * 0.1); // 前脚
      ctx.stroke();
    } else {
      // 「佇む (Stand)」の再現
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
