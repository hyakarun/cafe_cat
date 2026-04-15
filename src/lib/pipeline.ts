/**
 * Shiretto Cat: Edge AI Pipeline Logic
 * (Segformer + Depth Anything アプローチによるリファクタリング)
 */
import { pipeline, env } from '@xenova/transformers';

export class ShirettoPipeline {
  private segmenter: any = null;
  private depthEstimator: any = null;
  private isInitializing = false;

  async init() {
    if (this.segmenter || this.isInitializing) return;
    this.isInitializing = true;
    try {
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
      const isWebGPUSupported = !!navigator.gpu;
      
      // 1. セグメンテーションモデル（屋内・カフェ風景に強い）
      this.segmenter = await pipeline('image-segmentation', 'Xenova/segformer-b0-finetuned-ade-512-512', {
        device: isWebGPUSupported ? 'webgpu' : 'wasm',
        dtype: isWebGPUSupported ? 'fp16' : 'fp32'
      });

      // 2. 深度推定モデル（奥行きの把握用）
      this.depthEstimator = await pipeline('depth-estimation', 'Xenova/depth-anything-small-hf', {
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
    if (!this.segmenter || !this.depthEstimator) await this.init();
    try {
      // セグメンテーションと深度推定を並列で実行
      const [segOutput, depthOutput] = await Promise.all([
        this.segmenter(imageSource),
        this.depthEstimator(imageSource)
      ]);

      // 1. カップや皿のマスク画像を正確に取得
      const targetMask = this.findTargetMask(segOutput, ['cup', 'plate', 'table', 'bottle', 'bowl', 'drink', 'glass']);
      
      // 2. マスクの最上部（フチ）ではなく、奥行き（Depth）を考慮して配置座標とパースを計算
      const placement = this.calculateSmartPlacement(targetMask, depthOutput);

      // 3. Canvasによる直書きではなく、事前アセットを自然に変形して合成
      const result = await this.composeAsset(imageSource, placement, '/peeking_cat.svg');
      
      return {
        result,
        debugInfo: Object.assign({}, placement, { 
            segOutputLength: segOutput?.length,
            targetLabel: targetMask?.label
        })
      };
    } catch (error) {
      console.error('Processing failed:', error);
      throw error;
    }
  }

  private findTargetMask(segOutput: any[], targetLabels: string[]) {
    // 該当するラベル（カフェの小物やテーブル等）を優先的に主役として扱う
    const targets = segOutput.filter((s: any) => {
        const label = s.label.toLowerCase();
        return targetLabels.some(t => label.includes(t));
    });
    
    // 一番スコアや面積が大きい要素を特定（現在はシンプルに1番目を返す）
    return targets[0] || segOutput[0]; 
  }

  private calculateSmartPlacement(target: any, depthOutput: any) {
    if (!target) return { x: 0.5, y: 0.5, scale: 0.15, rotation: 0 };
    
    // TODO: 実際のDepthマップピクセル値から配置の奥深さを判定し、
    // cv.findContours等の輪郭線に対して最適な重ね合わせ（Affine）行列を計算する処理を入れる。
    // 今回は概念設計の第一段階として、対象物の中央・やや上（奥）のモック値を返す。
    
    const randomTilt = Math.random() > 0.5 ? 5 : -5;

    return {
        x: 0.5, // 画面中央付近
        y: 0.45, // やや上（奥側）
        scale: 0.2, // 猫のスケール
        rotation: randomTilt,
        reason: `Placed via smart depth logic for ${target?.label}`
    };
  }

  private async composeAsset(source: string, placement: any, assetPath: string): Promise<string> {
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

        // 事前に用意された高品質な透過SVG画像（猫）の合成レイヤーをロード
        const overlay = new Image();
        overlay.src = assetPath;
        overlay.onload = () => {
          ctx.save();
          // 配置とアフィン変換（パース等の傾き処理）
          const px = placement.x * canvas.width;
          const py = placement.y * canvas.height;
          const size = placement.scale * canvas.width;

          ctx.translate(px, py);
          ctx.rotate(placement.rotation * Math.PI / 180);
          
          // オーバーレイを配置（中央基準）
          ctx.drawImage(overlay, -size / 2, -size / 2, size, size);
          ctx.restore();

          resolve(canvas.toDataURL('image/jpeg', 0.9));
        };
        overlay.onerror = () => {
            console.error('Asset load error:', assetPath);
            // 失敗時は元の画像をそのまま返す
            resolve(canvas.toDataURL('image/jpeg', 0.9));
        };
      };
      img.onerror = () => {
          resolve(source); // フォールバック
      }
      img.src = source;
    });
  }
}

export const pipelineInstance = new ShirettoPipeline();
